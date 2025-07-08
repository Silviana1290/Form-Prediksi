"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Brain, Zap, BarChart3, TrendingUp, Target, Gauge } from "lucide-react"

interface ModelMetrics {
  accuracy: number
  mse: number
  mae: number
  r2: number
  trainingTime: number
  predictionSpeed: number
}

interface ModelComparisonProps {
  mlpMetrics: ModelMetrics
  svmMetrics: ModelMetrics
  ensembleMetrics: ModelMetrics
}

export function ModelComparisonChart({ mlpMetrics, svmMetrics, ensembleMetrics }: ModelComparisonProps) {
  const models = [
    {
      name: "MLP Neural Network",
      icon: Brain,
      color: "blue",
      metrics: mlpMetrics,
      description: "Deep learning with multiple hidden layers",
      strengths: ["Complex pattern recognition", "Non-linear relationships", "Feature interaction learning"],
      weaknesses: ["Longer training time", "Black box model", "Requires more data"],
    },
    {
      name: "SVM-RBF Kernel",
      icon: Zap,
      color: "green",
      metrics: svmMetrics,
      description: "Support Vector Machine with Radial Basis Function",
      strengths: ["Robust to outliers", "Good generalization", "Memory efficient"],
      weaknesses: ["Sensitive to feature scaling", "Limited interpretability", "Slower on large datasets"],
    },
    {
      name: "Ensemble Model",
      icon: BarChart3,
      color: "purple",
      metrics: ensembleMetrics,
      description: "Weighted combination of MLP (60%) and SVM (40%)",
      strengths: ["Best overall performance", "Reduced overfitting", "Combines model strengths"],
      weaknesses: ["Increased complexity", "Longer inference time", "Harder to interpret"],
    },
  ]

  const getColorClasses = (color: string) => {
    const colorMap = {
      blue: "text-blue-600 bg-blue-50 border-blue-200",
      green: "text-green-600 bg-green-50 border-green-200",
      purple: "text-purple-600 bg-purple-50 border-purple-200",
    }
    return colorMap[color as keyof typeof colorMap] || colorMap.blue
  }

  const getIconColorClasses = (color: string) => {
    const colorMap = {
      blue: "text-blue-500",
      green: "text-green-500",
      purple: "text-purple-500",
    }
    return colorMap[color as keyof typeof colorMap] || colorMap.blue
  }

  return (
    <div className="space-y-6">
      {/* Performance Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Model Performance Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {models.map((model, index) => {
              const IconComponent = model.icon
              return (
                <div key={index} className={`p-4 rounded-lg border ${getColorClasses(model.color)}`}>
                  <div className="flex items-center gap-2 mb-3">
                    <IconComponent className={`h-5 w-5 ${getIconColorClasses(model.color)}`} />
                    <h3 className="font-semibold">{model.name}</h3>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Accuracy</span>
                      <span className="font-bold">{model.metrics.accuracy.toFixed(1)}%</span>
                    </div>
                    <Progress value={model.metrics.accuracy} className="h-2" />
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-600">MAE: </span>
                        <span className="font-medium">{model.metrics.mae.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">R²: </span>
                        <span className="font-medium">{model.metrics.r2.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {models.map((model, index) => {
          const IconComponent = model.icon
          return (
            <Card key={index} className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <IconComponent className={`h-5 w-5 ${getIconColorClasses(model.color)}`} />
                  {model.name}
                </CardTitle>
                <p className="text-sm text-gray-600">{model.description}</p>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Metrics */}
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Accuracy</span>
                    <Badge variant="secondary">{model.metrics.accuracy.toFixed(1)}%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Mean Absolute Error</span>
                    <Badge variant="outline">{model.metrics.mae.toFixed(2)}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">R² Score</span>
                    <Badge variant="outline">{model.metrics.r2.toFixed(3)}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Training Time</span>
                    <Badge variant="outline">{model.metrics.trainingTime}min</Badge>
                  </div>
                </div>

                {/* Strengths */}
                <div>
                  <h4 className="font-medium text-green-700 mb-2 flex items-center gap-1">
                    <TrendingUp className="h-4 w-4" />
                    Strengths
                  </h4>
                  <ul className="text-sm space-y-1">
                    {model.strengths.map((strength, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-green-500 mt-1">•</span>
                        <span>{strength}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Weaknesses */}
                <div>
                  <h4 className="font-medium text-orange-700 mb-2 flex items-center gap-1">
                    <Gauge className="h-4 w-4" />
                    Considerations
                  </h4>
                  <ul className="text-sm space-y-1">
                    {model.weaknesses.map((weakness, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-orange-500 mt-1">•</span>
                        <span>{weakness}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Performance Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Model</th>
                  <th className="text-center p-2">Accuracy (%)</th>
                  <th className="text-center p-2">MSE</th>
                  <th className="text-center p-2">MAE</th>
                  <th className="text-center p-2">R² Score</th>
                  <th className="text-center p-2">Training Time</th>
                  <th className="text-center p-2">Prediction Speed</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => {
                  const IconComponent = model.icon
                  return (
                    <tr key={index} className="border-b hover:bg-gray-50">
                      <td className="p-2">
                        <div className="flex items-center gap-2">
                          <IconComponent className={`h-4 w-4 ${getIconColorClasses(model.color)}`} />
                          <span className="font-medium">{model.name}</span>
                        </div>
                      </td>
                      <td className="text-center p-2 font-medium">{model.metrics.accuracy.toFixed(1)}%</td>
                      <td className="text-center p-2">{model.metrics.mse.toFixed(2)}</td>
                      <td className="text-center p-2">{model.metrics.mae.toFixed(2)}</td>
                      <td className="text-center p-2">{model.metrics.r2.toFixed(3)}</td>
                      <td className="text-center p-2">{model.metrics.trainingTime}min</td>
                      <td className="text-center p-2">{model.metrics.predictionSpeed}ms</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Best Model Recommendation */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
        <CardHeader>
          <CardTitle className="text-green-800">🏆 Recommended Model</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-start gap-4">
            <BarChart3 className="h-8 w-8 text-purple-500 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-bold text-lg text-purple-700">Ensemble Model</h3>
              <p className="text-gray-700 mb-3">
                The ensemble model combining MLP (60%) and SVM (40%) provides the best overall performance with{" "}
                {ensembleMetrics.accuracy.toFixed(1)}% accuracy and {ensembleMetrics.r2.toFixed(3)} R² score.
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge className="bg-purple-100 text-purple-800">Best Accuracy</Badge>
                <Badge className="bg-purple-100 text-purple-800">Lowest MAE</Badge>
                <Badge className="bg-purple-100 text-purple-800">Highest R²</Badge>
                <Badge className="bg-purple-100 text-purple-800">Production Ready</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
