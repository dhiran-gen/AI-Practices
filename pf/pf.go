package pf

import (
	"fmt"

	"github.com/go-gota/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// Load data from CSV file
	df := dataframe.ReadCSV("stock_data.csv")

	// Split data into features and target
	features := base.NewDenseInstancesFromMat(df.Subset([]int{0, 1, 2, 3, 4, 5, 6, 7}).ToFloat64(), df.Nrow())
	target := base.NewDenseInstancesFromMat(df.Subset([]int{8}).ToFloat64(), df.Nrow())

	// Split data into training and testing sets
	trainData, testData := base.InstancesTrainTestSplit(features, 0.8)
	trainTarget, testTarget := base.InstancesTrainTestSplit(target, 0.8)

	// Create a linear regression model and train it on the training data
	lr := linear_models.NewLinearRegression()
	lr.Fit(trainData, trainTarget)

	// Evaluate the model on the testing data
	predictions, _ := lr.Predict(testData)
	r2, err := evaluation.RSquared(testTarget, predictions)
	if err != nil {
		panic(err)
	}

	// Print the R-squared score
	fmt.Println(r2)
}
