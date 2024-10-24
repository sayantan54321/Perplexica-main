import { readFileSync } from 'fs';
import path from 'path';

// import embeddingData from '../../final_embeddings.json'; // Path to your JSON file
// import summaryData from '../../final_summaries.json';

// const embeddingDataDict: Record<string, any> = embeddingData;
// const summaryDataDict: Record<string, string> = summaryData;

let embeddingDataDict: { [key: string]: number[] } = {};
let summaryDataDict: { [key: string]: string } = {};

try {
  const embeddingsPath = path.join(__dirname, '../../final_embeddings.json');
  const summariesPath = path.join(__dirname, '../../final_summaries.json');

  embeddingDataDict = JSON.parse(readFileSync(embeddingsPath, 'utf8'));
  summaryDataDict = JSON.parse(readFileSync(summariesPath, 'utf8'));
} catch (error) {
  console.warn(
    'Warning: Cache files not found, proceeding without cache:',
    error,
  );
  embeddingDataDict = {};
  summaryDataDict = {};
}

function getId(id: string) {
  id = id.replace(/\\/g, '/');
  return 'Knowledge/' + id.substring(id.search('Knowledge/'));
}

//knowledge/C:/Users/bhawe/OneDrive/Desktop/Gika/Search/Knowledge/40/puff sleeve maternity.md

export function getSummary(id: string) {
  try {
    id = getId(id);
    console.log('Fetching summary for: ', id);
  } catch (error) {
    console.error('Error getting summary:', error, '\n', 'For: ', id);
  }
  return summaryDataDict[id];
}
export function getEmbedding(id: string) {
  try {
    console.log(
      'Doc Embedding:',
      typeof embeddingDataDict['Knowledge/Knowledge/26/mesh maxi dress.md'],
      embeddingDataDict['Knowledge/Knowledge/26/mesh maxi dress.md'].length,
    );
    id = getId(id);
    console.log('Fetching Embedding for: ', id);
  } catch (error) {
    console.error('Error getting Embedding:', error, '\n', 'For: ', id);
  }
  return undefined;
  // return embeddingDataDict[id];
}
