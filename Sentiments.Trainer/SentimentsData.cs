using System;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentsData
{
    [LoadColumn(0)]
    public string? Text { get; set; }
    
    [LoadColumn(1)]
    public string? Score { get; set; }
}