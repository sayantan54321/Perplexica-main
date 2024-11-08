import { BaseMessage } from '@langchain/core/messages';
import handleWritingAssistant from './writingAssistant';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import axios from 'axios';

const FLASK_API_URL = "http://localhost:5000/find_products";

type UnifiedQueryInput = {
    query: string;
    chat_history: BaseMessage[];
    llm: BaseChatModel;
    embeddings: Embeddings;
};

const handleQueryForImages = async (
    input: UnifiedQueryInput
): Promise<{ attributes: Array<{ [key: string]: string }>, images: string[] }> => {
    const { query, chat_history, llm, embeddings } = input;
    
    console.log('Starting handleQueryForImages with query:', query);
    
    try {
        const extractedAttributes: Array<{ [key: string]: string }> = await new Promise((resolve, reject) => {
            console.log('Creating writing assistant emitter...');
            const emitter = handleWritingAssistant(query, chat_history, llm, embeddings);
            let attributesData = '';

            emitter.on('data', (data) => {
                console.log('Received data from emitter:', data);
                try {
                    const parsedData = JSON.parse(data);
                    if (parsedData.type === 'response') {
                        attributesData += parsedData.data;
                    }
                } catch (error) {
                    console.error('Error parsing emitter data:', error);
                }
            });

            emitter.on('end', () => {
                console.log('Emitter ended. Raw attributes data:', attributesData);
                try {
                    const formattedAttributes = JSON.parse(attributesData).flatMap((pairGroup: Array<{ [key: string]: string }>) => {
                        return pairGroup.map((pair) => ({ ...pair }));
                    });
                    console.log('Formatted attributes:', formattedAttributes);
                    resolve(formattedAttributes);
                } catch (error) {
                    console.error('Error formatting attributes:', error);
                    reject('Failed to parse attribute pairs');
                }
            });

            emitter.on('error', (error) => {
                console.error('Emitter error:', error);
                reject(error);
            });
        });

        const apiPayload = {
            input: JSON.stringify(extractedAttributes),
            min_match_threshold: 1
        };
        
        console.log('Sending request to Flask API with payload:', apiPayload);

        const response = await axios.post(FLASK_API_URL, apiPayload, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: 10000 // 10 second timeout
        });

        console.log('Received response from Flask API:', response.data);

        if (response.data && response.data.images) {
            return {
                attributes: extractedAttributes,
                images: response.data.images
            };
        } else {
            throw new Error('No images returned from the Flask API.');
        }
    } catch (error) {
        if (axios.isAxiosError(error)) {
            console.error('Axios error details:', {
                message: error.message,
                response: error.response?.data,
                status: error.response?.status,
                config: {
                    url: error.config?.url,
                    method: error.config?.method,
                    headers: error.config?.headers,
                    data: error.config?.data
                }
            });
        } else {
            console.error('Non-Axios error:', error);
        }
        throw error;
    }
};

export default handleQueryForImages;