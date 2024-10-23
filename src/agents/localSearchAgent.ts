import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';
import LineOutputParser from '../lib/outputParsers/lineOutputParser';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import { ChatOpenAI } from '@langchain/openai';
import { searchDocs, index_name } from '../utils/localElasticSearch';

// Question rephrasing prompt
const questionRephrasingPrompt = `
You are an AI question rephraser. You will be given a conversation and a follow-up question. Your task is to rephrase the follow-up question so it is a standalone question that can be used to search a local document database.
If it is a simple writing task or a greeting (unless the greeting contains a question after it) like Hi, Hello, How are you, etc., then you need to return \`not_needed\` as the response.
You must always return the rephrased question inside the \`question\` XML block.

<examples>
1. Follow up question: What is the capital of France?
Rephrased question:
<question>
Capital of France
</question>

2. Hi, how are you?
Rephrased question:
<question>
not_needed
</question>

3. Follow up question: Can you explain Docker in simple terms?
Rephrased question:
<question>
Explain Docker in simple terms
</question>
</examples>

Anything below is part of the actual conversation. Use the conversation and the follow-up question to rephrase the follow-up question as a standalone question based on the guidelines shared above.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
`;

// Elasticsearch response prompt
const elasticsearchResponsePrompt = `
You are Perplexica, an AI model expert at answering queries based on local document storage. 
Generate a response that is informative and relevant to the user's query based on provided context from our Elasticsearch index.
Use an unbiased and journalistic tone in your response. Do not repeat the text verbatim.
Your responses should be medium to long in length, informative, and relevant to the user's query. Use markdown to format your response and bullet points to list information.
Cite your sources using [number] notation at the end of each relevant sentence. The number refers to the document number in the provided context.
If you can't find relevant information, say "I'm sorry, I couldn't find any relevant information on this topic in our local documents. Would you like me to search for something else?"

<context>
{context}
</context>

Anything between the \`context\` tags is retrieved from our Elasticsearch index and is not part of the conversation with the user. Today's date is ${new Date().toISOString()}
`;

// Document summarization prompt
const documentSummarizationPrompt = `
You are a text summarizer. You need to summarize the text provided inside the \`text\` XML block. 
Summarize the text into 1 or 2 sentences capturing the main idea of the text.
Make sure that you don't miss any crucial points while summarizing the text.
You will also be given a \`query\` XML block which contains the user's query. Try to answer the query in the summary from the text provided.
If the query says "Summarize" then just summarize the text without specifically answering the query.
Only return the summarized text without any other messages, text or XML blocks.

<query>
{query}
</query>

<text>
{text}
</text>
`;

const strParser = new StringOutputParser();

// Search Elasticsearch function
const searchElasticsearch = async (query: string, index: string) => {
  return await searchDocs(query, index);
};

// Handle stream function
const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type ElasticsearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createQuestionRephrasingChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(questionRephrasingPrompt),
    llm,
    strParser,
    new LineOutputParser({ key: 'question' }),
    (question: string) => {
      if (question === 'not_needed') {
        return { query: '', docs: [] };
      }
      return { query: question };
    },
  ]);
};

const createElasticsearchRetrieverChain = (llm: BaseChatModel) => {
  (llm as unknown as ChatOpenAI).temperature = 0;

  return RunnableSequence.from([
    createQuestionRephrasingChain(llm),
    RunnableLambda.from(async (input: { query: string }) => {
      if (!input.query) return { query: '', docs: [] };

      const results = await searchElasticsearch(input.query, index_name);

      const documents = results.map(
        (doc) =>
          new Document({
            pageContent: doc.content ? doc.content : doc.title,
            metadata: {
              title: doc.title,
              url: doc.path, // This matches the web search structure where URL is used for source linking
            },
          }),
      );

      // Group documents by URL (filepath) like in web search
      const docGroups: Document[] = [];

      documents.map((doc) => {
        const URLDocExists = docGroups.find(
          (d) =>
            d.metadata.url === doc.metadata.url && d.metadata.totalDocs < 10,
        );

        if (!URLDocExists) {
          docGroups.push({
            ...doc,
            metadata: {
              ...doc.metadata,
              totalDocs: 1,
            },
          });
        }

        const docIndex = docGroups.findIndex(
          (d) =>
            d.metadata.url === doc.metadata.url && d.metadata.totalDocs < 10,
        );

        if (docIndex !== -1) {
          docGroups[docIndex].pageContent =
            docGroups[docIndex].pageContent + `\n\n` + doc.pageContent;
          docGroups[docIndex].metadata.totalDocs += 1;
        }
      });

      // Summarize grouped documents
      let docs = [];
      await Promise.all(
        docGroups.map(async (doc) => {
          const res = await llm.invoke(`
          You are a document summarizer, tasked with summarizing a piece of text retrieved from local documents. Your job is to summarize the 
          text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
          If the query is "summarize", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.
          
          - **Professional tone**: The summary should sound professional, not too casual or vague.
          - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
          - **Not too lengthy, but detailed**: The summary should be informative but not excessively long.

          <query>
          ${input.query}
          </query>

          <text>
          ${doc.pageContent}
          </text>
        `);

          const document = new Document({
            pageContent: res.content as string,
            metadata: {
              title: doc.metadata.title,
              url: doc.metadata.url, // This preserves the file path
            },
          });
          console.warn(
            document.metadata.title,
            ':\n',
            document.pageContent,
            '\n\n',
            document.metadata.url,
          );
          docs.push(document);
        }),
      );
      return { query: input.query, docs: docs };
    }),
  ]);
};

const createElasticsearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const elasticsearchRetrieverChain = createElasticsearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const [docEmbeddings, queryEmbedding] = await Promise.all([
      embeddings.embedDocuments(docs.map((doc) => doc.pageContent)),
      embeddings.embedQuery(query),
    ]);

    const similarity = docEmbeddings.map((docEmbedding, i) => ({
      index: i,
      similarity: computeSimilarity(queryEmbedding, docEmbedding),
    }));

    return similarity
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5)
      .map((sim) => docs[sim.index]);
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: ElasticsearchChainInput) => input.query,
      chat_history: (input: ElasticsearchChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        elasticsearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', elasticsearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const elasticsearchSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const elasticsearchAnsweringChain = createElasticsearchAnsweringChain(
      llm,
      embeddings,
    );

    const stream = elasticsearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({
        data: 'An error occurred while searching local documents. Please try again later.',
      }),
    );
    logger.error(`Error in Elasticsearch search: ${err}`);
  }

  return emitter;
};

const handleLocalSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = elasticsearchSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleLocalSearch;
