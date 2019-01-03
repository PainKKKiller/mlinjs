require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const mnist = require('mnist-data');
const loadCSV = require('./load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');

function loadData() {
  const mnistData = mnist.training(0, 60000);

  const features = mnistData.images.values.map(img => {
    return _.flatMap(img);
  });

  const encodedLabels = mnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });
  return { features, labels: encodedLabels };
}

const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 80,
  batchSize: 500
});

regression.train();

const testmnistData = mnist.testing(0, 1500);

const testFeatures = testmnistData.images.values.map(img => {
  return _.flatMap(img);
});

const testLabels = testmnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testLabels);
console.log('Accuracy', accuracy);

plot({
  x: regression.costHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Cost'
});


/*let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
  converters: {
    mpg: value => {
      const mpg = parseFloat(value);
      if (mpg < 15) {
        return [1, 0, 0];
      } else if (mpg < 30) {
        return [0, 1, 0];
      } else {
        return [0, 0, 1];
      }
    }
  }
});

console.log(_.flatMap(labels));

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10
});

regression.train();
const r2 = regression.test(testFeatures, _.flatMap(testLabels));
console.log('R2 is', r2);

console.log(regression.predict([
  [130, 307, 1.75]
]).print());

plot({
  x: regression.costHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Cost'
}); */

/* const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});

console.log('R2 is', r2);

regression.predict([[120, 2, 380]]).print(); */
