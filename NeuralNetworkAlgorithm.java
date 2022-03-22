import java.io.FileReader;
import java.util.Scanner;

public class NeuralNetworkAlgorithm {
//    hidden neurons h
    public static int h = 105;

//    Initialising the range of weights
    public static double max = 0.1;
    public static double min = -0.1;

//    Learning rate LR
    public static double LR = 0.0401;

//    Weights between input layer and hidden layer WH
    public static double[][] WH = new double[h][64];

//    Weights between hidden layer and output layer WO
    public static double[][] WO = new double[10][h];

//    cycles to run
    public static int Cycles = 300;

//    map for the desired outputs
    public static double[] targetOutput = new double[10];

//    Array to store the inputs (one row from the dataset)
    public static double[] dataSample = new double[64];

//    success counter
    public static int success = 0;

//    Storing outputs of Hidden neurons
    public static double[] outputHidden = new double[h];
    public static double[] outputNeuron = new double[10];

    public static void main(String[] args) {

//        initialising the weights.
        WH = assignWeights();
        WO = assignOutputWeights();

//        Running the training
        runNeuralNetwork();

//        Running the test method
        testing();
    }

//    The method that trains the algorithm
    public static void runNeuralNetwork(){


//        Printing the report
        System.out.println("---------------------Success Report-------------------\n" + " Iterations: " + Cycles +
                "\n Max and Min: (" + max + ", " + min + ")\n" + " Number of Hidden Neurons: " + h +
                "\n Number of Outer Neurons: 10" + "\n Learning Rate: " + LR);
        System.out.println("\n---------------------Training Set---------------------");

//        Counter to count the iterations
        int counter = 0;
        double temp_success = 0; //variable to store temporary success rate
        double acc = 0; //Variable to store the accuracy

//        Beginning of the iterations
        while(counter < Cycles){
            try{

//                Reading from file
                FileReader testFileReader = new FileReader("trainingSet.txt");
                Scanner reader = new Scanner(testFileReader);
                int target = 0;
                while(reader.hasNextLine()){
                    initializeOutput(); //clearing the desired outputs to all 0s
                    String fileLine = reader.nextLine();
                    String[] data = fileLine.split(",");
                    for(int i = 0; i < data.length; i++){
                        if(i < (data.length - 1 ) )
                            dataSample[i] = Double.parseDouble(data[i]);
                        else{
                            target = Integer.parseInt(data[i]);
                            targetOutput[target] = 1;
                        }
                    }
                    feedForward();

//                    Checking if there is error
                    if(testError()){
                        training();
                    }
                    else{
                        success++;
                    }
                }
                counter ++;

//                Calculating the success rate
                temp_success = success;
                acc = (temp_success / 2810) * 100 ;
                if (counter <= 5 ) {
                    System.out.println("Cycle: "+ counter+ " | Success: " + success +"/2810 | Accuracy: " + acc + " %");
                }
                else if(counter == 10 || counter == 25 || counter == 50 || counter == 100 || counter == 150 || counter == 200 || counter == 250 || counter == 300){
                    System.out.println("Cycle: "+ counter+ " | Success: " + success +"/2810 | Accuracy: " + acc + " %");
                }
                temp_success = 0;
                success = 0;

            }catch (Exception e){
                System.out.println("Exception: " + e);
            }

        }
    }


    public static void feedForward(){

//        Output of hidden layer neurons
        for(int i = 0; i < h; i++){
            double weighted_sum = 0;

//            Summation = Input * weights
            for(int j = 0; j < dataSample.length; j++){
                weighted_sum += dataSample[j] * WH[i][j];
            }
//            applying the sigmoid function
            double sigmoid_value = 1/(1 + Math.pow(2.71, -weighted_sum));
            outputHidden[i] = sigmoid_value;
        }

//        Output of output layer neurons
        for(int i = 0; i < outputNeuron.length; i++){
            double weighted_sum_op = 0;
            for(int j = 0; j < outputHidden.length; j++){
                weighted_sum_op += outputHidden[j] * WO[i][j];
            }

//            applying the threshold function
            if(weighted_sum_op >= 0){
                outputNeuron[i] = 1;
            }
            else{
                outputNeuron[i] = 0;
            }
        }

    }

//    Back Tracking method
    public static void training(){
        double[] errorOutput = new double[10];
        double[] errorHidden = new double[h];

//      Calculating error for output layer -> desired - actual
        for(int i = 0; i < errorOutput.length; i++){
            errorOutput[i] = targetOutput[i] - outputNeuron[i];
        }

//      Summation of the error for the hidden layer
        for(int j = 0; j < h; j++){
            double errorTemp = 0;
            for(int i = 0; i < outputNeuron.length; i++){
                errorTemp += errorOutput[i] * WO[i][j];
            }

//            Derivative of the Sigmoid function
            errorHidden[j] = outputHidden[j] * (1 - outputHidden[j] ) * errorTemp;
        }

//        adjusting Weights of Output layer WO
        for(int i = 0; i < outputNeuron.length; i++){
            for(int j = 0; j < h; j++){
                WO[i][j] = WO[i][j] + (LR * outputHidden[j] * errorOutput[i]);
            }
        }

//        adjusting Weights of Hidden Layer WH
        for(int i = 0; i < h; i++){
            for(int j = 0; j < dataSample.length; j++){
                WH[i][j] = WH[i][j] + (LR * dataSample[j] * errorHidden[i]);
            }
        }
    }

    public static boolean testError(){

//        Checking if the desired Output matches the actual output
        for(int i = 0; i < targetOutput.length; i++){
            if(targetOutput[i] != outputNeuron[i]){
                return true;
            }
        }
        return false;
    }


//    Testing Method
    public static void testing(){
        System.out.println("\n---------------------Testing Set---------------------");

        double success_rate = 0;
        try{

//            Reading from the test set
            FileReader testFileReader = new FileReader("testSet.txt");
            Scanner reader = new Scanner(testFileReader);
            int target = 0;
            while(reader.hasNextLine()){
                initializeOutput();
                String fileLine = reader.nextLine();
                String[] data = fileLine.split(",");
                for(int i = 0; i < data.length; i++){
                    if(i < (data.length - 1 ) )
                        dataSample[i] = Double.parseDouble(data[i]);
                    else{
                        target = Integer.parseInt(data[i]);
                        targetOutput[target] = 1;
                    }

                }
                feedForward();
                if(testError() == false){
                    success_rate++;
                }
            }
            double accuracy;
            accuracy = (success_rate / 2810) * 100;
            System.out.println("\nSuccess: " + success_rate+"/2810 | Accuracy: " + accuracy  + " %");

        }catch (Exception e){
            System.out.println("Exception: " + e);
        }

    }

//  Assigns random weights for the hidden layer
    public static double[][] assignWeights(){
        double [][] weights = new double [h][64];
        for (int outer = 0; outer < h ; outer ++){
            for ( int inner = 0; inner < 64; inner ++){
                weights[outer][inner] = (double) Math.round((Math.random() * (max - min) + min) * 100)/100;
            }
        }
        return weights;
    }

//    Assigns random weights for the output layer
    public static double[][] assignOutputWeights(){
        double [][] weights = new double [10][h];
        for (int outer = 0; outer < weights.length ; outer ++){
            for ( int inner = 0; inner < h; inner ++){
                weights[outer][inner] = (double) Math.round((Math.random() * (max - min) + min)*100)/100;
            }
        }
        return weights;
    }

//    Clears the target output array for every iteration
    public static void initializeOutput(){
        for(int a = 0; a < targetOutput.length; a++){
            targetOutput[a] = 0;
        }
    }
}
//End of Code
