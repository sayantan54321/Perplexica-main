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
import { getEmbedding, getSummary } from '../utils/precomputedData';

// Question rephrasing prompt
const questionRephrasingPrompt = `
You are an AI question rephraser. You will be given a conversation and a follow-up question. Your task is to rephrase the follow-up question so it is a clear, specific, and detailed standalone question that can be used to search a local document database.
Ensure that the rephrased question is formulated to elicit comprehensive information.
If it is a simple greeting or doesn't require information retrieval, return \`not_needed\` as the response.
You must always return the rephrased question inside the \`<question>\` XML block.
**Do NOT include any context-based phrases like "Based on the provided text" or similar. Start your question directly.**

<examples>
1. Follow-up question: What is the capital of France?
Rephrased question:
<question>
What is the capital city of France, and what are its major historical landmarks?
</question>

2. Hi, how are you?
Rephrased question:
<question>
not_needed
</question>

3. Follow-up question: Can you explain Docker in simple terms?
Rephrased question:
<question>
Provide a detailed explanation of Docker and how it simplifies application deployment.
</question>
</examples>

The following conversation and follow-up question are for you to use to rephrase the question.

<conversation>
{chat_history}
</conversation>

Follow-up question: {query}
Rephrased question:

`;


// Elasticsearch response prompt
const elasticsearchResponsePrompt = `
You are Perplexica, an AI model specialized in providing direct, relevant, and informative answers using data from our local Elasticsearch index.

**STRICT INSTRUCTIONS**:
- Provide thorough and detailed answers that fully address the user's query, consisting of multiple paragraphs as necessary.
- Do NOT use phrases like "Based on the provided text," "According to the context," or any similar introductory phrases. Start your response directly with the relevant information.
- If you are tempted to write any introductory phrase, STOP and rewrite your answer to start immediately with the factual content.
- Your tone should be professional, and your response must be formatted in markdown. Use bullet points or numbered lists to organize information if necessary.
- Cite sources using [number] notation, where the number corresponds to the document in the provided context.
- If no relevant information is found, respond with: "I'm sorry, I couldn't find any relevant information on this topic in our local documents. Would you like me to search for something else?"

**EXAMPLES OF INCORRECT RESPONSES** (Must NOT use these):
- "Based on the context provided, ..."
- "According to the given text, ..."
- "Based on the provided context..."
- "It appears that the information suggests ..."
      
**EXAMPLES OF CORRECT RESPONSES**:
- "The capital city of France is Paris, known for its rich history and cultural landmarks such as the Eiffel Tower and the Louvre Museum. [1]"
- "Docker is a platform that allows developers to package applications into containers—standardized executable components combining application source code with the operating system libraries and dependencies required to run that code in any environment. [2]"

<context>
{context}
</context>

Today's date is ${new Date().toISOString()}
**REMINDER**: Provide a comprehensive and detailed answer that fully addresses the user's query.

`;



// Document summarization prompt
const documentSummarizationPrompt = `
You are a text summarizer. You need to provide a comprehensive, multi-paragraph explanation of the text provided inside the \`text\` XML block.
Your explanation should capture all main ideas and thoroughly answer the user's query.
Make sure you don't miss any crucial points while explaining the text.
**Do not include sentences like 'Based on the provided context...' or 'Based on the conversation' in your response.**
**Do not give any introductory sentences; directly provide the detailed explanation.**
You will also be given a \`query\` XML block which contains the user's query. Answer the query thoroughly in your explanation using the text provided.
If the query says "Summarize," provide a comprehensive summary of the text.
Only return the explanation without any other messages, text, or XML blocks.
Do not include any phrases like 'Based on the context' or 'Based on the conversation.' Directly provide the explanation without additional introductions.

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

      const documents = results
        .filter(
          (doc) =>
            typeof doc.content === 'string' || typeof doc.title === 'string',
        )
        .map(
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
      // Process documents with cache awareness
      let docs = [];
      await Promise.all(
        docGroups.map(async (doc) => {
          let summaryContent;

          // Check if summary exists in cache
          if (getSummary(doc.metadata.url)) {
            // Get summary from cache
            summaryContent = getSummary(doc.metadata.url);
            console.warn('Using cached summary for:', doc.metadata.url);
          } else {
            // Generate new summary if not in cache
            const res = await llm.invoke(`
              You are a document summarizer tasked with providing comprehensive explanations based on the text provided. Your job is to produce a detailed, multi-paragraph response that captures all main ideas and thoroughly answers the query.
              Do not include sentences like 'Based on the provided context...' or 'Based on the conversation' in your response.
              Do not provide any introductory sentences; directly start with the detailed explanation.
              If the query is "summarize," provide a comprehensive summary of the text. If the query is a specific question, answer it thoroughly in your explanation.
              
              - **Professional tone**: The explanation should be professional and informative.
              - **Thorough and detailed**: Capture every key point from the text and ensure the explanation directly answers the query.
              - **Comprehensive**: Provide as much relevant information as needed, including examples and details.
              - **No introductory sentences**: Start directly with the key information.
            
              <query>
              ${input.query}
              </query>
            
              <text>
              ${doc.pageContent}
              </text>
            `);            
            summaryContent = res.content as string;
            console.warn('Generated new summary for:', doc.metadata.url);
          }

          const document = new Document({
            pageContent: summaryContent,
            metadata: {
              title: doc.metadata.title,
              url: doc.metadata.url,
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

    // Get query embedding
    const queryEmbedding = await embeddings.embedQuery(query);
    console.log(
      'Query Embedding:',
      typeof queryEmbedding,
      queryEmbedding.length,
    );

    // Process document embeddings with cache awareness
    const docEmbeddings = await Promise.all(
      docs.map(async (doc) => {
        if (getEmbedding(doc.metadata.url)) {
          console.warn('Using cached embedding for:', doc.metadata.url);
          return getEmbedding(doc.metadata.url);
        } else {
          console.warn('Generating new embedding for:', doc.metadata.url);
          const embedding = (
            await embeddings.embedDocuments([doc.pageContent])
          )[0];
          return embedding;
        }
      }),
    );

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
