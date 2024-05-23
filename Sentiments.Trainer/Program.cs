using System;
using Microsoft.ML;
using Microsoft.ML.Data;
namespace Sentiments.Trainer;

class Program
{
    static void Main(string[] args)
    {        
        var trainModel = false;
        var textInput = "I hated this movie! I have never seen anything worse!";
        var dataInputPath = "data/sentiments-test5k.csv";
        var modelOutputPath = "models/sentiments-test5k.zip";        
        
        var context = new MLContext();
        if(trainModel)
        {
            // 1. Load data

            var dataView = context.Data.LoadFromTextFile<SentimentsData>(dataInputPath, hasHeader: true, separatorChar: ',');

            // 2. Describe features and labels
            var dataProcessPipeline = context.Transforms.Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentsData.Text))
                .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(SentimentsData.Score)));

            // 3. Build training pipeline
            var trainingPipeline = dataProcessPipeline
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));
;
            
            // 4. Train model
            var model = trainingPipeline.Fit(dataView);
            
            // 5. Evaluate model
            var predictions = model.Transform(dataView);
            var metrics = context.MulticlassClassification.Evaluate(predictions, "Label");
            Console.WriteLine("MicroAccuracy: " + metrics.MicroAccuracy);
            Console.WriteLine("MacroAccuracy: " + metrics.MacroAccuracy);
            Console.WriteLine("LogLoss: " + metrics.LogLoss);
            Console.WriteLine("LogLossReduction: " + metrics.LogLossReduction);

            // 6. Save model to .zip file
            context.Model.Save(model, dataView.Schema, modelOutputPath);
        }

        // 7. Load the model
        var modelPath = Path.Combine(Directory.GetCurrentDirectory(), modelOutputPath);
        var loadedModel = context.Model.Load(modelPath, out var modelInputSchema);
        
        // 8. Use model to make prediction
        var predictor = context.Model.CreatePredictionEngine<SentimentsData, SentimentsPrediction>(loadedModel);
        var testData = new SentimentsData { Text = textInput};
        var prediction = predictor.Predict(testData);

        // Output the prediction results
        var index = GetIndexOfMaxValue(prediction.Score);
        var predictedLabel = MapPredictedLabel(index);
        Console.WriteLine("");
        Console.WriteLine($"You seem '{predictedLabel}' with your review.");
        Console.WriteLine("Scores: [" + string.Join(", ", prediction.Score) + "]");
    }

    private static string MapPredictedLabel(int predictedLabel)
    {
        switch (predictedLabel)
        {
            case 0:
                return "Negative";
            case 1:
                return "Neutrual";
            case 2:
                return "Positive";
            default:
                return "Unknown";
        }
    }

    public static int GetIndexOfMaxValue(float[] array)
    {
        if (array == null || array.Length == 0)
            throw new ArgumentException("Array cannot be null or empty.");

        int maxIndex = 0;
        float maxValue = array[0];

        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > maxValue)
            {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}