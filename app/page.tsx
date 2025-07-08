"use client"
import { useState, useEffect } from "react"

interface FormData {
  // Process Parameters
  temperature: string
  pressure: string
  materialType: string
  processTime: string

  // Material Properties
  materialDensity: string
  materialHardness: string
  materialComposition: string

  // Equipment Settings
  equipmentType: string
  machineSpeed: string
  toolWear: string
  maintenanceStatus: string

  // Environmental Conditions
  humidity: string
  ambientTemp: string
  vibrationLevel: string

  // Quality Control
  inspectionLevel: string
  toleranceLevel: string

  // Production Context
  batchSize: string
  operatorExperience: string
  shiftType: string

  // Additional Notes
  additionalNotes: string
}

interface PredictionResult {
  mlpScore: number
  svmScore: number
  ensembleScore: number
  confidence: number
  category: string
  recommendations: string[]
  processOptimization: string[]
  qualityInsights: string[]
  featureImportance: Record<string, number>
  predictionBreakdown: {
    temperatureImpact: number
    pressureImpact: number
    materialImpact: number
    operatorImpact: number
    equipmentImpact: number
  }
}

export default function ManufacturingPredictionForm() {
  const [formData, setFormData] = useState<FormData>({
    temperature: "",
    pressure: "",
    materialType: "",
    processTime: "",
    materialDensity: "",
    materialHardness: "",
    materialComposition: "",
    equipmentType: "",
    machineSpeed: "",
    toolWear: "",
    maintenanceStatus: "",
    humidity: "",
    ambientTemp: "",
    vibrationLevel: "",
    inspectionLevel: "",
    toleranceLevel: "",
    batchSize: "",
    operatorExperience: "",
    shiftType: "",
    additionalNotes: "",
  })

  const [showHelp, setShowHelp] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [predictionHistory, setPredictionHistory] = useState<PredictionResult[]>([])

  // Auto-calculate derived metrics
  useEffect(() => {
    const temp = Number.parseFloat(formData.temperature) || 0
    const pressure = Number.parseFloat(formData.pressure) || 0

    if (temp > 0 && pressure > 0) {
      // These calculations will be used in the ML models
      const tempPressureProduct = temp * pressure
      const materialFusionMetric = temp * pressure * 0.8
      const materialTransformationMetric = temp ** 2 * pressure

      console.log(`Derived Metrics:`)
      console.log(`Temperature x Pressure: ${tempPressureProduct}`)
      console.log(`Material Fusion Metric: ${materialFusionMetric}`)
      console.log(`Material Transformation Metric: ${materialTransformationMetric}`)
    }
  }, [formData.temperature, formData.pressure])

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  // Enhanced MLP Model Simulation
  const calculateMLPScore = (data: FormData): number => {
    let score = 50 // Base score

    // Temperature factor with non-linear activation
    const temp = Number.parseFloat(data.temperature) || 0
    const tempNormalized = (temp - 165) / 15 // Normalize around optimal
    const tempActivation = Math.tanh(tempNormalized) // Tanh activation
    score += tempActivation * 15

    // Pressure factor with ReLU-like activation
    const pressure = Number.parseFloat(data.pressure) || 0
    const pressureNormalized = (pressure - 25) / 5
    const pressureActivation = Math.max(0, 1 - Math.abs(pressureNormalized)) // ReLU-like
    score += pressureActivation * 12

    // Temperature-Pressure interaction (hidden layer simulation)
    const tempPressureInteraction = temp * pressure
    if (tempPressureInteraction >= 3000 && tempPressureInteraction <= 5000) {
      score += 10
    } else if (tempPressureInteraction >= 2500 && tempPressureInteraction < 3000) {
      score += 6
    }

    // Material type with learned weights
    const materialWeights = {
      steel: 8.5,
      aluminum: 6.2,
      plastic: 4.1,
      composite: 9.8,
      ceramic: 7.3,
    }

    if (data.materialType && materialWeights[data.materialType as keyof typeof materialWeights]) {
      score += materialWeights[data.materialType as keyof typeof materialWeights]
    }

    // Equipment and operator factors (deep learning patterns)
    const maintenanceWeights = {
      excellent: 12,
      good: 6,
      fair: -2,
      poor: -8,
    }

    const operatorWeights = {
      expert: 10,
      experienced: 6,
      intermediate: 2,
      novice: -4,
    }

    if (data.maintenanceStatus && maintenanceWeights[data.maintenanceStatus as keyof typeof maintenanceWeights]) {
      score += maintenanceWeights[data.maintenanceStatus as keyof typeof maintenanceWeights]
    }

    if (data.operatorExperience && operatorWeights[data.operatorExperience as keyof typeof operatorWeights]) {
      score += operatorWeights[data.operatorExperience as keyof typeof operatorWeights]
    }

    // Environmental factors
    const humidity = Number.parseFloat(data.humidity) || 50
    const humidityOptimal = 1 - Math.abs(humidity - 50) / 50
    score += humidityOptimal * 5

    // Process time optimization
    const processTime = Number.parseFloat(data.processTime) || 45
    if (processTime >= 30 && processTime <= 60) {
      score += 4
    }

    return Math.max(0, Math.min(100, Math.round(score)))
  }

  // Enhanced SVM Model Simulation with RBF Kernel
  const calculateSVMScore = (data: FormData): number => {
    let score = 55 // Different base score for SVM

    const temp = Number.parseFloat(data.temperature) || 0
    const pressure = Number.parseFloat(data.pressure) || 0

    /* --- RBF kernel distance from optimal operating point --- */
    const optimalTemp = 165
    const optimalPressure = 25
    const gamma = 0.1

    const distance = Math.pow((temp - optimalTemp) / 15, 2) + Math.pow((pressure - optimalPressure) / 5, 2)

    const rbfValue = Math.exp(-gamma * distance)
    score += rbfValue * 20

    /* --- Support-vector boundary checks on Temp/Pressure ratio --- */
    const tempPressureRatio = temp / (pressure || 1)
    if (tempPressureRatio >= 5.5 && tempPressureRatio <= 7.5) {
      score += 15
    } else if (tempPressureRatio >= 4.5 && tempPressureRatio < 5.5) {
      score += 8
    } else if (tempPressureRatio > 7.5 && tempPressureRatio <= 8.5) {
      score += 5
    } else {
      score -= 5
    }

    /* --- Material influence (categorical) --- */
    const materialFactors = {
      steel: 6,
      aluminum: 4,
      plastic: 3,
      composite: 7,
      ceramic: 5,
    }
    if (data.materialType && materialFactors[data.materialType as keyof typeof materialFactors]) {
      score += materialFactors[data.materialType as keyof typeof materialFactors]
    }

    /* --- Maintenance impact --- */
    const maintenanceFactors = {
      excellent: 8,
      good: 4,
      fair: -1,
      poor: -6,
    }
    if (data.maintenanceStatus && maintenanceFactors[data.maintenanceStatus as keyof typeof maintenanceFactors]) {
      score += maintenanceFactors[data.maintenanceStatus as keyof typeof maintenanceFactors]
    }

    /* --- Operator experience impact --- */
    const operatorFactors = {
      expert: 6,
      experienced: 4,
      intermediate: 1,
      novice: -3,
    }
    if (data.operatorExperience && operatorFactors[data.operatorExperience as keyof typeof operatorFactors]) {
      score += operatorFactors[data.operatorExperience as keyof typeof operatorFactors]
    }

    /* --- Humidity penalty (farther from 50 % → lower score) --- */
    const humidity = Number.parseFloat(data.humidity) || 50
    score -= Math.abs(humidity - 50) * 0.05

    /* --- Clamp to 0-100 and return --- */
    return Math.max(0, Math.min(100, Math.round(score)))
  }
}
