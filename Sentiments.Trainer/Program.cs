using System;
using Microsoft.ML;
using Microsoft.ML.Data;
namespace Sentiments.Trainer;

class Program
{
    static void Main(string[] args)
    {        
        var trainModel = true;
        var textInput = "This was really good!";
        var dataInputPath = "data/sentiments-debug1k.csv";
        var modelOutputPath = "models/sentiments-debug.zip";        
        
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
            context.Model.Save(model, dataView.Schema, Constants.ModelPath);
        }

        // 7. Load the model
        var modelPath = Path.Combine(Directory.GetCurrentDirectory(), modelOutputPath);
        var loadedModel = context.Model.Load(modelPath, out var modelInputSchema);
        
        // 8. Use model to make prediction
        var predictor = context.Model.CreatePredictionEngine<SentimentsData, SentimentsPrediction>(loadedModel);
        var testData = new SentimentsData { Text = textInput};
        var prediction = predictor.Predict(testData);
        Console.WriteLine($"Predicted: {prediction.PredictedLabel}");
    }
}