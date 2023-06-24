import { CSVLoader } from "langchain/document_loaders/fs/csv";

export const run = async () => {

const loader = new CSVLoader(
  "/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.csv",
  "text"
);

const docs = await loader.load();

console.log(docs);
  // const loader = new CSVLoader("/Users/tedshaffer/Documents/Projects/ai/tsLcKb/src/example.csv");

  // const docs = await loader.load();
  // console.log(docs);

};
