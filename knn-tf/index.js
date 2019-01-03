require('@tensorflow/tfjs-node');

require('@tensorflow/tfjs-node-gpu');

const tf = require('@tensorflow/tfjs');

const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV('./kc_house_data.csv', {
            shuffle: true,
            splitTest: 10,
            dataColumns: ['lat', 'long', 'sqft_lot'],
            labelColumns: ['price']
        });

console.log(testFeatures, testLabels);


function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
    return features
            .sub(mean)
            .div(variance.pow(0.5))
            .pow(2)
            .sum(1)
            .pow(0.5)
            .expandDims(1)
            .concat(labels, 1)
            .unstack()
            .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
            .slice(0, k)
            .reduce((acc, pair) => acc + pair.get(1), 0) / k;
}

// console.log(features);
let f = tf.tensor(features);
let l = tf.tensor(labels);

console.log('f', f.shape);
console.log('l', l.shape);

testFeatures.forEach((testPoint, i) => {
    const result = knn(f, l, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('error: ', err * 100);
})

const result = knn(f, l, tf.tensor(testFeatures[0]), 10);
const err = testLabels[0][0] - 

console.log('guess', result, testLabels[0][0])

