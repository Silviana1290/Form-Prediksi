"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Download, FileText, Table, BarChart3, Settings } from "lucide-react"

interface ExportData {
  predictions: any[]
  modelMetrics: any
  processParameters: any
  qualityAnalysis: any
}

interface ExportFunctionalityProps {
  data: ExportData
}

export function ExportFunctionality({ data }: ExportFunctionalityProps) {
  const [exportFormat, setExportFormat] = useState<string>("")
  const [exportType, setExportType] = useState<string>("")
  const [isExporting, setIsExporting] = useState(false)

  const exportOptions = [
    {
      id: "predictions",
      name: "Prediction Results",
      description: "Export all prediction results with model comparisons",
      icon: BarChart3,
      formats: ["CSV", "Excel", "JSON"],
    },
    {
      id: "metrics",
      name: "Model Performance Metrics",
      description: "Export detailed model performance and evaluation metrics",
      icon: Table,
      formats: ["PDF", "Excel", "JSON"],
    },
    {
      id: "report",
      name: "Complete Analysis Report",
      description: "Comprehensive report with visualizations and insights",
      icon: FileText,
      formats: ["PDF", "Word"],
    },
    {
      id: "parameters",
      name: "Process Parameters",
      description: "Export process parameters and optimization recommendations",
      icon: Settings,
      formats: ["CSV", "Excel", "PDF"],
    },
  ]

  const handleExport = async () => {
    if (!exportType || !exportFormat) {
      alert("Please select export type and format")
      return
    }

    setIsExporting(true)

    try {
      // Simulate export process
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Generate export data based on type and format
      const exportData = generateExportData(exportType, exportFormat)

      // Create and download file
      downloadFile(exportData, `manufacturing_${exportType}_${Date.now()}.${exportFormat.toLowerCase()}`)

      alert(`Successfully exported ${exportType} as ${exportFormat}`)
    } catch (error) {
      alert("Export failed. Please try again.")
    } finally {
      setIsExporting(false)
    }
  }

  const generateExportData = (type: string, format: string) => {
    switch (type) {
      case "predictions":
        return generatePredictionsExport(format)
      case "metrics":
        return generateMetricsExport(format)
      case "report":
        return generateReportExport(format)
      case "parameters":
        return generateParametersExport(format)
      default:
        return ""
    }
  }

  const generatePredictionsExport = (format: string) => {
    const predictions = data.predictions || []

    if (format === "CSV") {
      const headers = [
        "ID",
        "Temperature",
        "Pressure",
        "Material",
        "MLP_Score",
        "SVM_Score",
        "Ensemble_Score",
        "Quality_Category",
        "Confidence",
      ]
      const rows = predictions.map((pred, index) => [
        index + 1,
        pred.temperature || 0,
        pred.pressure || 0,
        pred.material || "Unknown",
        pred.mlpScore || 0,
        pred.svmScore || 0,
        pred.ensembleScore || 0,
        pred.category || "Unknown",
        pred.confidence || 0,
      ])

      return [headers, ...rows].map((row) => row.join(",")).join("\n")
    }

    if (format === "JSON") {
      return JSON.stringify(
        {
          exportDate: new Date().toISOString(),
          totalPredictions: predictions.length,
          predictions: predictions,
          summary: {
            averageQuality: predictions.reduce((sum, p) => sum + (p.ensembleScore || 0), 0) / predictions.length,
            highQualityCount: predictions.filter((p) => (p.ensembleScore || 0) >= 85).length,
            modelAgreement: predictions.filter((p) => Math.abs((p.mlpScore || 0) - (p.svmScore || 0)) < 10).length,
          },
        },
        null,
        2,
      )
    }

    return "Export data generated"
  }

  const generateMetricsExport = (format: string) => {
    const metrics = data.modelMetrics || {}

    if (format === "JSON") {
      return JSON.stringify(
        {
          exportDate: new Date().toISOString(),
          modelPerformance: {
            mlp: {
              accuracy: 85.3,
              mse: 52.8,
              mae: 6.3,
              r2: 0.847,
              trainingTime: "15 minutes",
            },
            svm: {
              accuracy: 83.2,
              mse: 51.2,
              mae: 6.1,
              r2: 0.832,
              trainingTime: "8 minutes",
            },
            ensemble: {
              accuracy: 86.3,
              mse: 49.5,
              mae: 5.9,
              r2: 0.863,
              trainingTime: "23 minutes",
            },
          },
          crossValidation: {
            folds: 5,
            averageAccuracy: 84.7,
            standardDeviation: 2.1,
          },
          featureImportance: {
            temperature: 0.18,
            pressure: 0.16,
            materialType: 0.14,
            processTime: 0.12,
            operatorExperience: 0.1,
          },
        },
        null,
        2,
      )
    }

    return "Model metrics export data"
  }

  const generateReportExport = (format: string) => {
    const reportContent = `
# Manufacturing Quality Prediction Analysis Report

## Executive Summary
This report presents the results of machine learning-based quality prediction analysis for manufacturing processes using Multi-Layer Perceptron (MLP) and Support Vector Machine (SVM) models.

## Model Performance
- **Best Model**: Ensemble (MLP + SVM)
- **Overall Accuracy**: 86.3%
- **Mean Absolute Error**: 5.9
- **R² Score**: 0.863

## Key Findings
1. Temperature and pressure are the most critical parameters
2. Ensemble model provides best prediction accuracy
3. Material type significantly impacts quality outcomes
4. Operator experience correlates with quality scores

## Recommendations
1. Maintain temperature between 150-180°C for optimal quality
2. Keep pressure in 20-30 kPa range
3. Implement real-time monitoring system
4. Focus on operator training programs

## Technical Details
- **Training Dataset**: 1000 samples
- **Validation Method**: 5-fold cross-validation
- **Feature Engineering**: 11 key features identified
- **Model Optimization**: Grid search with hyperparameter tuning

Generated on: ${new Date().toLocaleDateString()}
    `

    return reportContent
  }

  const generateParametersExport = (format: string) => {
    if (format === "CSV") {
      const headers = ["Parameter", "Optimal_Min", "Optimal_Max", "Current_Avg", "Recommendation"]
      const rows = [
        ["Temperature (°C)", "150", "180", "165.2", "Within optimal range"],
        ["Pressure (kPa)", "20", "30", "25.1", "Within optimal range"],
        ["Process Time (min)", "30", "60", "45.3", "Consider optimization"],
        ["Humidity (%)", "40", "60", "52.1", "Monitor closely"],
        ["Batch Size", "50", "200", "148.7", "Optimal size"],
      ]

      return [headers, ...rows].map((row) => row.join(",")).join("\n")
    }

    return "Process parameters export data"
  }

  const downloadFile = (content: string, filename: string) => {
    const blob = new Blob([content], { type: "text/plain" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-5 w-5" />
          Export Analysis Results
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Export Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {exportOptions.map((option) => {
            const IconComponent = option.icon
            return (
              <div
                key={option.id}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  exportType === option.id ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => setExportType(option.id)}
              >
                <div className="flex items-start gap-3">
                  <IconComponent className="h-5 w-5 text-blue-500 mt-1" />
                  <div className="flex-1">
                    <h3 className="font-medium">{option.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">{option.description}</p>
                    <div className="flex gap-1 mt-2">
                      {option.formats.map((format) => (
                        <Badge key={format} variant="outline" className="text-xs">
                          {format}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Format Selection */}
        {exportType && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Select Export Format</label>
              <Select value={exportFormat} onValueChange={setExportFormat}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose format" />
                </SelectTrigger>
                <SelectContent>
                  {exportOptions
                    .find((opt) => opt.id === exportType)
                    ?.formats.map((format) => (
                      <SelectItem key={format} value={format}>
                        {format}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>

            {/* Export Button */}
            <Button onClick={handleExport} disabled={!exportFormat || isExporting} className="w-full">
              {isExporting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Export {exportType} as {exportFormat}
                </>
              )}
            </Button>
          </div>
        )}

        {/* Export History */}
        <div className="border-t pt-4">
          <h3 className="font-medium mb-3">Recent Exports</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Prediction Results (CSV)</span>
              <Badge variant="secondary">2 hours ago</Badge>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Model Metrics (PDF)</span>
              <Badge variant="secondary">1 day ago</Badge>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Complete Report (PDF)</span>
              <Badge variant="secondary">3 days ago</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
