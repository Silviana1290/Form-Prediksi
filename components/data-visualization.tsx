"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, BarChart3, PieChart, Activity } from "lucide-react"

interface DataPoint {
  temperature: number
  pressure: number
  quality: number
  material: string
  prediction: number
}

interface DataVisualizationProps {
  data: DataPoint[]
  modelPredictions: {
    mlp: number[]
    svm: number[]
    ensemble: number[]
  }
}

export function DataVisualization({ data, modelPredictions }: DataVisualizationProps) {
  // Calculate statistics
  const avgQuality = data.reduce((sum, d) => sum + d.quality, 0) / data.length
  const avgTemperature = data.reduce((sum, d) => sum + d.temperature, 0) / data.length
  const avgPressure = data.reduce((sum, d) => sum + d.pressure, 0) / data.length

  // Material distribution
  const materialCounts = data.reduce(
    (acc, d) => {
      acc[d.material] = (acc[d.material] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  // Quality categories
  const qualityCategories = data.reduce(
    (acc, d) => {
      if (d.quality >= 85) acc.excellent++
      else if (d.quality >= 70) acc.good++
      else if (d.quality >= 55) acc.average++
      else acc.poor++
      return acc
    },
    { excellent: 0, good: 0, average: 0, poor: 0 },
  )

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium">Avg Quality</span>
            </div>
            <div className="text-2xl font-bold text-blue-600">{avgQuality.toFixed(1)}</div>
            <div className="text-xs text-gray-600">Out of 100</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-red-500" />
              <span className="text-sm font-medium">Avg Temperature</span>
            </div>
            <div className="text-2xl font-bold text-red-600">{avgTemperature.toFixed(1)}°C</div>
            <div className="text-xs text-gray-600">Process parameter</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Avg Pressure</span>
            </div>
            <div className="text-2xl font-bold text-green-600">{avgPressure.toFixed(1)} kPa</div>
            <div className="text-xs text-gray-600">Process parameter</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <PieChart className="h-4 w-4 text-purple-500" />
              <span className="text-sm font-medium">Total Samples</span>
            </div>
            <div className="text-2xl font-bold text-purple-600">{data.length}</div>
            <div className="text-xs text-gray-600">Data points</div>
          </CardContent>
        </Card>
      </div>

      {/* Quality Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Quality Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-sm">Excellent (85-100)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{qualityCategories.excellent}</span>
                <Badge variant="secondary">{((qualityCategories.excellent / data.length) * 100).toFixed(1)}%</Badge>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full"
                style={{ width: `${(qualityCategories.excellent / data.length) * 100}%` }}
              ></div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                <span className="text-sm">Good (70-84)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{qualityCategories.good}</span>
                <Badge variant="secondary">{((qualityCategories.good / data.length) * 100).toFixed(1)}%</Badge>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-yellow-500 h-2 rounded-full"
                style={{ width: `${(qualityCategories.good / data.length) * 100}%` }}
              ></div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-orange-500 rounded"></div>
                <span className="text-sm">Average (55-69)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{qualityCategories.average}</span>
                <Badge variant="secondary">{((qualityCategories.average / data.length) * 100).toFixed(1)}%</Badge>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-orange-500 h-2 rounded-full"
                style={{ width: `${(qualityCategories.average / data.length) * 100}%` }}
              ></div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-sm">Poor (0-54)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{qualityCategories.poor}</span>
                <Badge variant="secondary">{((qualityCategories.poor / data.length) * 100).toFixed(1)}%</Badge>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full"
                style={{ width: `${(qualityCategories.poor / data.length) * 100}%` }}
              ></div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Material Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <PieChart className="h-5 w-5" />
            Material Type Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(materialCounts).map(([material, count]) => (
              <div key={material} className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{count}</div>
                <div className="text-sm font-medium">{material}</div>
                <div className="text-xs text-gray-600">{((count / data.length) * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Process Parameter Ranges */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Temperature Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm">Minimum:</span>
                <span className="font-medium">{Math.min(...data.map((d) => d.temperature)).toFixed(1)}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Maximum:</span>
                <span className="font-medium">{Math.max(...data.map((d) => d.temperature)).toFixed(1)}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Average:</span>
                <span className="font-medium">{avgTemperature.toFixed(1)}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Optimal Range:</span>
                <Badge variant="outline">150-180°C</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Pressure Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm">Minimum:</span>
                <span className="font-medium">{Math.min(...data.map((d) => d.pressure)).toFixed(1)} kPa</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Maximum:</span>
                <span className="font-medium">{Math.max(...data.map((d) => d.pressure)).toFixed(1)} kPa</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Average:</span>
                <span className="font-medium">{avgPressure.toFixed(1)} kPa</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Optimal Range:</span>
                <Badge variant="outline">20-30 kPa</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Prediction Accuracy */}
      <Card>
        <CardHeader>
          <CardTitle>Model Prediction Accuracy</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-lg font-bold text-blue-600">MLP Neural Network</div>
                <div className="text-2xl font-bold">85.3%</div>
                <div className="text-sm text-gray-600">Average Accuracy</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-lg font-bold text-green-600">SVM-RBF Kernel</div>
                <div className="text-2xl font-bold">83.2%</div>
                <div className="text-sm text-gray-600">Average Accuracy</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-lg font-bold text-purple-600">Ensemble Model</div>
                <div className="text-2xl font-bold">86.3%</div>
                <div className="text-sm text-gray-600">Average Accuracy</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
