import express from 'express';
import handleQueryForImages from '../agents/handleQueryforImages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { getAvailableChatModelProviders } from '../lib/providers';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import logger from '../utils/logger';

const router = express.Router();

router.post('/', async (req, res) => {
  console.log('Received request body:', req.body);
  
  try {
    let { query, chat_history, chat_model_provider, chat_model } = req.body;
    
    if (!query) {
      return res.status(400).json({ message: 'Query is required' });
    }

    console.log('Processing chat history...');
    chat_history = chat_history.map((msg: any) => {
      console.log('Processing message:', msg);
      if (msg.role === 'user') {
        return new HumanMessage(msg.content);
      } else if (msg.role === 'assistant') {
        return new AIMessage(msg.content);
      }
    });

    console.log('Getting chat models...');
    const chatModels = await getAvailableChatModelProviders();
    const provider = chat_model_provider ?? Object.keys(chatModels)[0];
    const chatModel = chat_model ?? Object.keys(chatModels[provider])[0];

    console.log('Selected model:', { provider, chatModel });

    let llm: BaseChatModel | undefined;

    if (chatModels[provider] && chatModels[provider][chatModel]) {
      llm = chatModels[provider][chatModel].model as BaseChatModel;
    }

    if (!llm) {
      logger.error('Invalid LLM model configuration:', { provider, chatModel });
      return res.status(500).json({ message: 'Invalid LLM model selected' });
    }

    console.log('Calling handleQueryForImages...');
    const result = await handleQueryForImages({
      query,
      chat_history,
      llm,
      embeddings: {} as any 
    });

    console.log('Got result:', result);

    res.status(200).json({ 
      attributes: result.attributes, 
      images: result.images 
    });
  } catch (err: any) {
    console.error('Full error:', err);
    
    const errorDetails = {
      message: err.message,
      stack: err.stack,
      response: err.response?.data,
      status: err.response?.status,
      config: err.config ? {
        url: err.config.url,
        method: err.config.method,
        headers: err.config.headers,
        data: err.config.data
      } : undefined
    };

    logger.error('Error in image search:', errorDetails);
    
    res.status(500).json({ 
      message: 'An error occurred while processing your request',
      details: process.env.NODE_ENV === 'development' ? errorDetails : undefined
    });
  }
});

export default router;