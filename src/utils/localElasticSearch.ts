import { Client } from '@elastic/elasticsearch';

// Define an interface for our Markdown document
interface MarkdownDoc {
  filename: string;
  content: string;
  last_modified: Date;
}

// Create a client instance
const client = new Client({ node: 'http://localhost:9200' });

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
      .filter((hit) => hit._source !== undefined) // Ensure _source is defined
      .map((hit) => ({
        ...(hit._source as MarkdownDoc),
        last_modified: new Date(hit._source!.last_modified), // Parse 'last_modified' to a Date object
      }));
  } catch (error) {
    console.error('Error searching documents:', error);
    return [];
  }
}
