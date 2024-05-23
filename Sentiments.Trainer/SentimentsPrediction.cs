using System;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentsPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; }
}