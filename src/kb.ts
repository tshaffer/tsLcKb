import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { BufferMemory } from "langchain/memory";
import * as fs from "fs";
import { VectorStoreRetriever } from "langchain/dist/vectorstores/base";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from 'langchain/chat_models/openai';

// export const run = async () => {

//   // const chat = new ChatOpenAI({});

//   /* Initialize the LLM to use to answer the question */
//   const model = new ChatOpenAI(
//     {
//       modelName: 'gpt-3.5-turbo',
//     }
//   );
//   /* Load in the file we want to do question answering over */
//   const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/state_of_the_union.txt", "utf8");
//   /* Split the text into chunks */
//   const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
//   const docs = await textSplitter.createDocuments([text]);

//   const embeddings = new OpenAIEmbeddings();

//   const db: FaissStore = await FaissStore.fromDocuments(docs, embeddings);

//   const retriever: VectorStoreRetriever<FaissStore> = db.asRetriever();
  
//   const chain: RetrievalQAChain = RetrievalQAChain.fromLLM(model, retriever);

//   const res = await chain.call({
//     // query: "What did the president say about Justice Breyer?",
//     query: "What is the capital city of France??",
//   });
//   console.log({ res });

// };

// export const run = async () => {
//   /* Initialize the LLM to use to answer the question */
//   const model = new OpenAI({});
//   /* Load in the file we want to do question answering over */
//   const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/state_of_the_union.txt", "utf8");
//   /* Split the text into chunks */
//   const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
//   const docs = await textSplitter.createDocuments([text]);
//   /* Create the vectorstore */
//   const vectorStore = await FaissStore.fromDocuments(docs, new OpenAIEmbeddings());
//   /* Create the chain */
//   const chain = ConversationalRetrievalQAChain.fromLLM(
//     model,
//     vectorStore.asRetriever(),
//     {
//       memory: new BufferMemory({
//         memoryKey: "chat_history", // Must be set to "chat_history"
//       }),
//     }
//   );
//   /* Ask it a question */
//   const question = "What did the president say about Justice Breyer?";
//   const res = await chain.call({ question });
//   console.log(res);
//   /* Ask it a follow up question */
//   const followUpRes = await chain.call({
//     question: "Was that nice?",
//   });
//   console.log(followUpRes);
// };



// import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
// import { Document } from "langchain/document";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// import { TextLoader } from "langchain/document_loaders/fs/text";

// import { FaissStore } from "langchain/vectorstores/faiss";

// import * as fs from "fs";
// import { VectorStoreRetriever } from "langchain/dist/vectorstores/base";

// import { RetrievalQAChain } from "langchain/chains";
// import { ChatOpenAI } from "langchain/dist/chat_models/openai";

// export const run = async () => {
//   // Initialize the LLM to use to answer the question.
//   const model = new OpenAI({});
//   const text = fs.readFileSync("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/state_of_the_union.txt", "utf8");
//   const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
//   const docs = await textSplitter.createDocuments([text]);

//   // Create a vector store from the documents.
//   const vectorStore = await FaissStore.fromDocuments(docs, new OpenAIEmbeddings());

//   // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
//   const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
//   const res = await chain.call({
//     query: "What did the president say about Justice Breyer?",
//   });
//   console.log({ res });
//   /*
//   {
//     res: {
//       text: 'The president said that Justice Breyer was an Army veteran, Constitutional scholar,
//       and retiring Justice of the United States Supreme Court and thanked him for his service.'
//     }
//   }
//   */
// };

export const run = async () => {
  try {

    const llmA = new OpenAI({});
    const chainA = loadQAStuffChain(llmA);

    // const loader = new TextLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt");
    // const docs = await loader.load();

    //split text into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 150,
      chunkOverlap: 20,
    });

    const text = fs.readFileSync(
      require.resolve("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt"),
      "utf8"
    );

    const docs = await textSplitter.createDocuments([text])

    const resA = await chainA.call({
      input_documents: docs,
      question: "Where is the rat poop broom?",
    });
    console.log({ resA });

  } catch (error) {
    console.log("error", error);
  }
};

// export const run = async () => {
//   try {

//     const llmA = new OpenAI({});
//     const chainA = loadQAStuffChain(llmA);

//     // const loader = new TextLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt");
//     // const docs = await loader.load();

//     //split text into chunks
//     const textSplitter = new RecursiveCharacterTextSplitter({
//       chunkSize: 150,
//       chunkOverlap: 20,
//     });

//     const text = fs.readFileSync(
//       require.resolve("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt"),
//       "utf8"
//     );

//     const docs = await textSplitter.createDocuments([text])

//     const resA = await chainA.call({
//       input_documents: docs,
//       question: "Where is the rat poop broom?",
//     });
//     console.log({ resA });

//   } catch (error) {
//     console.log("error", error);
//   }
// };

// export const run = async () => {
//   try {

//     const llmA = new OpenAI({});

//     //split text into chunks
//     const textSplitter = new RecursiveCharacterTextSplitter({
//       chunkSize: 150,
//       chunkOverlap: 20,
//     });

//     const text = fs.readFileSync(
//       require.resolve("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt"),
//       "utf8"
//     );

//     const docs = await textSplitter.createDocuments([text])
//     console.log('Number of documents created from splitter: ', docs.length);

//     console.log('Preview');
//     console.log(docs[0].pageContent + '\n');
//     console.log(docs[1].pageContent + '\n');

//     const embeddings = new OpenAIEmbeddings();

//     const db: FaissStore = await FaissStore.fromDocuments(docs, embeddings);

//     const retriever: VectorStoreRetriever<FaissStore> = db.asRetriever();
//     // const query = 'Where are the paper towels?';
//     // const query = 'Where does Lori store paper towels?';
//     const query = 'Where is the rat poop broom?';
//     const answers: Document[] = await retriever.getRelevantDocuments(query);
//     console.log('answers');
//     console.log(answers);

//     const similarity = await db.similaritySearch(query, 1);
//     console.log('similarity');
//     console.log(similarity);
//     console.log(similarity[0].metadata);

//   } catch (error) {
//     console.log("error", error);
//   }
// };





// Create a new index from texts
// import { FaissStore } from "langchain/vectorstores/faiss";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";

// export const run = async () => {
//   const vectorStore = await FaissStore.fromTexts(
//     ["Hello world", "Bye bye", "hello nice world"],
//     [{ id: 2 }, { id: 1 }, { id: 3 }],
//     new OpenAIEmbeddings()
//   );

//   const resultOne = await vectorStore.similaritySearch("hello world", 1);
//   console.log(resultOne);
// };

// Create a new index from a loader
// import { FaissStore } from "langchain/vectorstores/faiss";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// import { TextLoader } from "langchain/document_loaders/fs/text";

// export const run = async () => {
//   // Create docs with a loader
//   const loader = new TextLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.txt");
//   const docs = await loader.load();

//   // Load the docs into the vector store
//   const vectorStore = await FaissStore.fromDocuments(
//     docs,
//     new OpenAIEmbeddings()
//   );

//   // Search for the most similar document
//   const resultOne = await vectorStore.similaritySearch("hello world", 1);
//   console.log(resultOne);
// };

// Save an index to file and load it again
// import { FaissStore } from "langchain/vectorstores/faiss";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";

// export const run = async () => {

//   // Create a vector store through any method, here from texts as an example
//   const vectorStore = await FaissStore.fromTexts(
//     ["Hello world", "Bye bye", "hello nice world"],
//     [{ id: 2 }, { id: 1 }, { id: 3 }],
//     new OpenAIEmbeddings()
//   );

//   // Save the vector store to a directory
//   const directory = "/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src";

//   await vectorStore.save(directory);

//   // Load the vector store from the same directory
//   const loadedVectorStore = await FaissStore.load(
//     directory,
//     new OpenAIEmbeddings()
//   );

//   // vectorStore and loadedVectorStore are identical
//   const result = await loadedVectorStore.similaritySearch("hello world", 1);
//   console.log(result);
// }