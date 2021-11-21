package com.example.dl4j.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;

/**
 * A single-layer feedforward network for classifying MNIST images.
 *
 * @author Kevin Lee
 */
public class MultiLayerPerceptron {

    private final static Logger log = LoggerFactory.getLogger(MultiLayerPerceptron.class);

    public static void main(String[] args) throws Exception {
        final int numberOfRows = 28;
        final int numberOfCols = 28;
        final int numberOfLabels = 10;
        final int numberOfEpochs = 1;
        final int seed = 0;
        final int batchSize = 64;
        final double momentum = 0.9;
        final double learningRate = 0.001;
        final File file = new File("./weights/mnist/mlp");

        // Initialize data sets
        log.info("Initializing datasets");

        DataSetIterator trainSetIterator = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator testSetIterator = new MnistDataSetIterator(batchSize, false, seed);

        // Initialize model network
        MultiLayerNetwork model;

        if (file.exists()) {
            log.info("Loading model from '" + file.getPath() + "'");

            model = MultiLayerNetwork.load(file, true);
        }
        else {
            log.info("Building model from scratch");

            MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs.Builder()
                            .learningRate(learningRate)
                            .momentum(momentum).build())
                    .regularization(List.of(new L1Regularization(learningRate * 0.005)))
                    .list().layer(0, new DenseLayer.Builder()
                            .nIn(numberOfRows * numberOfCols)
                            .nOut(1000)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(1000)
                            .nOut(numberOfLabels)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .backpropType(BackpropType.Standard)
                    .build();

            model = new MultiLayerNetwork(configuration);
        }

        model.init();
        model.setListeners(new ScoreIterationListener(50));

        // Train network model
        log.info("Training network");

        for (int epoch = 1; epoch <= numberOfEpochs; epoch++) {
            log.info("Epoch " + epoch);

            model.fit(trainSetIterator);
        }

        // Test network model
        log.info("Evaluating network");

        Evaluation evaluation = new Evaluation(numberOfLabels);

        while (testSetIterator.hasNext()) {
            DataSet dataSet = testSetIterator.next();

            evaluation.eval(
                    dataSet.getLabels(), model.output(dataSet.getFeatures()));
        }

        // Saving model weights
        log.info("Saving model to '" + file.getPath() + "'");

        model.save(file);
    }

}
