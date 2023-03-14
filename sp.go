package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// Load data from CSV file
	data, err := base.ParseCSVToInstances("spData - Sheet1.csv", true)
	if err != nil {
		panic(err)
	}

	// Split data into training and testing sets
	trainData, testData := base.InstancesTrainTestSplit(data, 0.8)

	// Create a k-NN classifier and train it on the training data
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)
	cls.Fit(trainData)

	// Evaluate the model on the testing data
	predictions, _ := cls.Predict(testData)
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(err)
	}

	// Print the confusion matrix
	fmt.Println(confusionMat)
}
