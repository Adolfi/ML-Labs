using System;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentsPrediction
{
    [ColumnName("PredictedLabel")]
    public float PredictedLabel { get; set; }

    public float[] Score { get; set; }
}