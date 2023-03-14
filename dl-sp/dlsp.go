package dlsp

import (
    "bufio"
    "encoding/csv"
    "fmt"
    "io"
    "log"
    "math/rand"
    "os"
    "strconv"
    "strings"
    "time"

    "gorgonia.org/gorgonia"
    "gorgonia.org/tensor"
)

func main() {
    // Load data from CSV file
    file, err := os.Open("stock_data.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    reader.FieldsPerRecord = 9
    rawData, err := reader.ReadAll()
    if err != nil {
        log.Fatal(err)
    }

    // Convert data to float64 matrix
    data := make([]float64, 0)
    for _, record := range rawData {
        for _, value := range record {
            f, err := strconv.ParseFloat(value, 64)
            if err != nil {
                log.Fatal(err)
            }
            data = append(data, f)
        }
    }

    rows := len(rawData)
    cols := len(rawData[0])
    X := tensor.New(tensor.WithShape(rows, cols-1), tensor.WithBacking(data[:rows*(cols-1)]))
    y := tensor.New(tensor.WithShape(rows, 1), tensor.WithBacking(data[rows*(cols-1):]))

    // Split data into training and testing sets
    Xtrain, Xtest, ytrain, ytest := splitData(X, y, 0.7)

    // Define neural network model
    g := gorgonia.NewGraph()
    W1 := randomWeight(g, 8, cols-1)
    b1 := randomBias(g, 8)
    W2 := randomWeight(g, 1, 8)
    b2 := randomBias(g, 1)

    x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(rows, cols-1), gorgonia.WithName("x"))
    y_ := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(rows, 1), gorgonia.WithName("y"))
    hidden := gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, W1)), b1))))
    output := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(hidden, W2)), b2))

    // Define loss function and optimization algorithm
    cost := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(output, y_))))))
    solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(50), gorgonia.WithLearnRate(0.001), gorgonia.WithEpsilon(1e-5))

    // Train the model
    for i := 0; i < 1000; i++ {
        shuffleData(Xtrain, ytrain)
        if err := gorgonia.Learn(g, cost, solver, Xtrain, ytrain); err != nil {
            log.Fatal(err)
        }
    }

    // Evaluate the model on the testing data
    ypred, err := predict(g, Xtest)
    if err != nil {
        log.Fatal(err)
    }

    // Compute the R-squared score for the predictions
    r2 := rSquared
