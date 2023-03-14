package mlsp

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/gonum/matrix/mat64"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// Load data from CSV file
	file, err := os.Open("stock_data.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 9
	rawData, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	// Convert data to float64 matrix
	data := make([]float64, 0)
	for _, record := range rawData {
		for _, value := range record {
			f, err := strconv.ParseFloat(value, 64)
			if err != nil {
				panic(err)
			}
			data = append(data, f)
		}
	}

	rows := len(rawData)
	cols := len(rawData[0])
	X := mat64.NewDense(rows, cols-1, data[:rows*(cols-1)])
	y := mat64.NewDense(rows, 1, data[rows*(cols-1):])

	// Split data into training and testing sets
	Xtrain, Xtest, ytrain, ytest := base.InstancesTrainTestSplit(X, y, 0.7)

	// Create a linear regression model and train it on the training data
	reg := linear_models.NewLinearRegression()
	reg.Fit(Xtrain, ytrain)

	// Evaluate the model on the testing data
	yhat := reg.Predict(Xtest)
	r2 := rSquared(yhat, ytest)
	fmt.Printf("R-squared: %.4f\n", r2)
}

// Compute the R-squared score for two matrices of predicted and actual values
func rSquared(yhat *mat64.Dense, y *mat64.Dense) float64 {
	yhatMean := mean(yhat)
	ssResidual := sumOfSquares(y.Sub(yhat))
	ssTotal := sumOfSquares(y.Sub(yhatMean))
	return 1 - ssResidual/ssTotal
}

// Compute the mean of a matrix
func mean(m *mat64.Dense) float64 {
	var sum float64
	row, _ := m.Dims()
	for i := 0; i < row; i++ {
		sum += m.At(i, 0)
	}
	return sum / float64(row)
}

// Compute the sum of squares of a matrix
func sumOfSquares(m *mat64.Dense) float64 {
	var sum float64
	row, col := m.Dims()
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			sum += math.Pow(m.At(i, j), 2)
		}
	}
	return sum
}
