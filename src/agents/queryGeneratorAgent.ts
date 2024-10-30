import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { BaseMessage } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { ChatOpenAI } from '@langchain/openai';

// Attribute extraction prompt
const attributeExtractionPrompt = `
## EXTRACT_ATTRIBUTE_VALUE_PAIRS
You are an intelligent chatbot tasked with **accurately extracting key** \`attribute: value\` **pairs** from user queries. Your goal is to **identify and extract explicit** \`attribute: value\` **pairs** from the queries, strictly following the instructions below. You will also consider the context of previous queries when responding to subsequent ones. You should maintain the chat history with you until anyone asks you to reset it. If you get any minor spelling mistakes, try to give a response based on the correct spellings of that.

2. **Consider Context for Subsequent Queries**:
    - When processing a second (or subsequent) query, take the context of the previous query into account. Make sure that your response is consistent with attributes selected in the earlier query.

3. **Attributes to Extract**:
    - Focus on product attributes such as (Strictly limited to):
        'Brand', 'Category', 'Product_category', 'Subcategory', 'Size', 'Native_color', 'Primary_color', 'Pattern', 'Type', 'Product_type', 'Color_family', 'Dress_type', 'Fabric', 'Neckline', 'Occasion', 'Fit', 'Color', 'Sleeve_style', 'Neck', 'Length', 'Closure', 'Sleeve_length', 'Print', 'Waist_detail', 'Style', 'Model_size', 'Strap_style', 'Native_neck', 'Feature', 'Embellishment', 'Detail', 'Hemline', 'Skirt_style', 'Native_brand', 'Trim', 'Collection_name', 'Line', 'Pocket', 'Collar_style', 'Fabric_detail', 'Cuff_style', 'Origin', 'Length_dress', 'Hem_feature_shape', 'Bust', 'Care_instruction', 'Sheerness_semi', 'Print_type_pattern', 'Fit_description_size', 'Composition', 'Height_model_range', 'Measurement_size_length', 'Bust_model_size', 'Hip_model_size', 'Hip', 'Size_worn_run', 'Undergarment','Accent', 'Silhouette', 'Design', 'Fasten', 'Decoration', 'Gender', 'Bodice_style', 'Thigh_leg_style', 'Review_fabric', 'Review_closure', 'Sustainable', 'Production_method_manufacturing', 'Runway_season_collection', 'Fabric_description', 'Waistband', 'Hem_style', 'Tiering_style', 'Make_in', 'Belt_buckle', 'Edge', 'Stretch_non_stretchiness', 'Collaboration', 'Slit', 'Embroidery', 'Texture', 'Weave_construction', 'Seam', 'Smock_panel', 'Bodycon_style_body', 'Review_neck', 'Pocket_drop', 'Body_line', 'Pad_chest', 'Mesh_type_panel', 'Wash_cleaning_method', 'Import', 'Weight_feel', 'Bra', 'Shell_fabric', 'Hem_front_back', 'Measurement'.

4. **Inclusion of attribute Category**:
    - One attribute you must include in your response is \`category\`.

5. **Re-mapping of extracted attributes**:
    - You must map the extracted attributes to the attributes given. Be wise and make sure you map your extracted attributes to the nearest attributes given. You must not include any attributes that are not defined in the given attributes above.

6. **Consistency in Naming**:
    - Ensure consistency in naming attributes and normalize them when necessary:
        - Example: \`short sleeves\` → \`"Sleeve_length: Short"\`
        - Example: \`blue\` → \`"Colour: Blue"\`

7. **Maintain Language**:
    - The extracted \`attribute: value\` pairs must be in the same language as the provided query.

8. **Don't miss any important keywords**:
    - Never miss any important keywords from the given query and map them with the correct attribute given above.

9. **Maintain Consistency in attribute and keywords**:
    - Don't include unnecessary attributes/keywords in the output unless you are asked to include.

10. **Separation of multiple values belonging to the same attribute**:
    - If multiple values belong to the same attribute, separate them like this (\`"attribute":"value1"\`),(\`"attribute":"value2"\`)... and so on.

11. **Output Format**:
    - Each pair should be formatted as a list of strings: (\`"attribute": "value"\`)
    - Final output should be formatted strictly like this only: [("attribute1": "value1"), ("attribute2": "value2"),...]
    - No extra line also don't give chat history in your response.
    - Provide the output as a pure list of \`attribute: value\` pairs without any additional commentary or explanation.

Let's begin. Now answer the following query(s).
Query: {query}
`;

type AttributeExtractorInput = {
  query: string;
  chat_history: BaseMessage[];
  attribute: string;
};

// Helper function to map and extract the attributes
const createAttributeExtractorChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: AttributeExtractorInput) => input.query,
      attribute: (input: AttributeExtractorInput) => input.attribute,
      chat_history: (input: AttributeExtractorInput) =>
        input.chat_history.map(message => message.content).join("\n"),
    }),
    PromptTemplate.fromTemplate(attributeExtractionPrompt),
    llm,
  ]);
};

const extractAttributes = (
  input: AttributeExtractorInput,
  llm: BaseChatModel,
) => {
  (llm as unknown as ChatOpenAI).temperature = 0;
  const attributeExtractorChain = createAttributeExtractorChain(llm);
  return attributeExtractorChain.invoke(input);
};

export default extractAttributes;
