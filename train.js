const tf = require('@tensorflow/tfjs-node');
const use = require('@tensorflow-models/universal-sentence-encoder');
const fs = require('fs');

async function runChatbot() {
  console.log('Loading model...');
  const model = await use.load();
  console.log('Model loaded.');
  // json parse
  const greetings = require("./data/greetings.json").greetings;
   // define prompt
   var prompt = require('prompt-sync')();



  while (true) {
    const input = prompt('You: ');
    const embeddings = await model.embed(input.toLowerCase());
    
    // check if input contains user's name
    const name = getName(input);
    if (name) {
      console.log(`Bot: Hi ${name}! Nice to meet you.`);
    } else {
      const distances = await getDistances(embeddings, greetings, model);
      const closestIndex = tf.argMin(distances).dataSync()[0];

      if (distances[closestIndex] < 0.7) {
        console.log(`Bot: Hello! I am Stacy.`);
      } else {
        console.log('Bot: Sorry, I did not understand that.');
      } }

  }
}
function getName(input) {
    const words = input.toLowerCase().split(' ');
    if (words.includes('my') && words.includes('name') && words.includes('is') && words.length > 3) {
      return words[words.indexOf('is') + 1];
    } else {
      return null;
    }
  }
  
  async function getDistances(embeddings, greetings, model) {
    const promises = greetings.map(async (g) => {
      const embedding = await model.embed(g.toLowerCase());
      const distance = tf.norm(embeddings.sub(embedding), 'euclidean').dataSync()[0];
      return distance;
    });
    
  const distances = await Promise.all(promises);
  return distances;
}
runChatbot();