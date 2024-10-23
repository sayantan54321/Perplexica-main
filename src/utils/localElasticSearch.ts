import { Client } from '@elastic/elasticsearch';

// Define an interface for our Markdown document
interface MarkdownDoc {
  title: string;
  content: string;
  path: string;
}

// Create a client instance
const client = new Client({ node: 'http://host.docker.internal:9200' });

export const index_name = 'myntra';

// Function to search documents
export async function searchDocs(
  query: string,
  index_string: string,
): Promise<MarkdownDoc[]> {
  try {
    // Provide 'MarkdownDoc' as a generic to the search method
    const result = await client.search<MarkdownDoc>({
      index: index_string, // Elasticsearch 8.x no longer requires the type field
      body: {
        query: {
          multi_match: {
            query: query,
            fields: ['filename', 'content'], // Searching across 'filename' and 'content'
          },
        },
      },
    });

    // Now the hits will be typed correctly, adding a check for `_source`
    return result.hits.hits
      .filter((hit) => hit._source)
      .map((hit) => hit._source as MarkdownDoc);
  } catch (error) {
    console.error('Error searching documents:', error);
    return [];
  }
}
