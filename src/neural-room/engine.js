import { LEARNING_RATE, MAX_HISTORY } from './constants.js'

function cloneWeights(weights) {
  return weights.map((layerWeights) => layerWeights.map((row) => row.slice()))
}

function cloneLayers(values) {
  return values.map((layerValues) => layerValues.slice())
}

function createZeroWeightUpdates(weights) {
  return weights.map((layerWeights) => layerWeights.map((row) => row.map(() => 0)))
}

function multiplyVectorMatrix(vector, matrix) {
  return matrix[0].map((_, columnIndex) =>
    vector.reduce(
      (sum, value, rowIndex) => sum + value * matrix[rowIndex][columnIndex],
      0,
    ),
  )
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value))
}

function sigmoidVector(vector) {
  return vector.map((value) => sigmoid(value))
}

function subtractScaledMatrix(weights, updates, scalar) {
  return weights.map((layerWeights, layerIndex) =>
    layerWeights.map((row, rowIndex) =>
      row.map(
        (value, columnIndex) => value - scalar * updates[layerIndex][rowIndex][columnIndex],
      ),
    ),
  )
}

function outerProduct(left, right) {
  return left.map((leftValue) => right.map((rightValue) => leftValue * rightValue))
}

function buildTrace(weights, input, target, layers) {
  const zLayers = [input.slice()]
  const activations = [input.slice()]

  for (let layerIndex = 0; layerIndex < weights.length; layerIndex += 1) {
    const sums = multiplyVectorMatrix(activations[layerIndex], weights[layerIndex])
    zLayers.push(sums)
    activations.push(sigmoidVector(sums))
  }

  const prediction = activations[activations.length - 1]
  const sampleLoss =
    prediction.reduce((sum, value, index) => sum + Math.pow(target[index] - value, 2), 0) /
    prediction.length
  const deltas = layers.map((count) => Array.from({ length: count }, () => 0))
  const outputLayerIndex = layers.length - 1

  deltas[outputLayerIndex] = prediction.map(
    (value, index) => -2 * (target[index] - value) * value * (1 - value),
  )

  for (let layerIndex = layers.length - 2; layerIndex >= 1; layerIndex -= 1) {
    deltas[layerIndex] = activations[layerIndex].map((activation, nodeIndex) => {
      let error = 0

      for (let nextNodeIndex = 0; nextNodeIndex < deltas[layerIndex + 1].length; nextNodeIndex += 1) {
        error += deltas[layerIndex + 1][nextNodeIndex] * weights[layerIndex][nodeIndex][nextNodeIndex]
      }

      return error * activation * (1 - activation)
    })
  }

  const sampleUpdates = weights.map((_, layerIndex) =>
    outerProduct(activations[layerIndex], deltas[layerIndex + 1]),
  )

  return {
    input: input.slice(),
    target: target.slice(),
    prediction: prediction.slice(),
    sampleLoss,
    zLayers: cloneLayers(zLayers),
    activations: cloneLayers(activations),
    deltas: cloneLayers(deltas),
    sampleUpdates: cloneWeights(sampleUpdates),
    zeroUpdates: createZeroWeightUpdates(weights),
  }
}

function createMaskedActivations(trace, visibleLayerIndex, layers) {
  return layers.map((count, layerIndex) => {
    if (layerIndex <= visibleLayerIndex) {
      return trace.activations[layerIndex].slice()
    }

    return Array.from({ length: count }, () => 0)
  })
}

export class NeuralNetworkEngine {
  constructor(modelConfig) {
    this.modelConfig = modelConfig
    this.layers = modelConfig.layers.slice()
    this.trainingData = modelConfig.trainingData.map(({ input, target }) => ({
      input: input.slice(),
      target: target.slice(),
    }))
    this.initialWeights = cloneWeights(modelConfig.initialWeights)
    this.learningRate = LEARNING_RATE
    this.maxHistory = MAX_HISTORY
    this.generatedEpochs = []
    this.activations = this.layers.map((count) => Array.from({ length: count }, () => 0))
    this.biases = []
    this.weights = cloneWeights(this.initialWeights)
    this.lastInput = this.trainingData[0].input.slice()
    this.currentSnapshot = null
    this.currentPhase = 'ready'
    this.currentEpoch = 0
    this.pendingTrace = null
    this.forwardLayerIndex = 0
    this.backwardLayerIndex = null

    this.initialize()
  }

  initialize() {
    this.generatedEpochs = []
    this.weights = cloneWeights(this.initialWeights)
    this.lastInput = this.trainingData[0].input.slice()
    this.biases = []
    this.currentPhase = 'ready'
    this.currentEpoch = 0
    this.pendingTrace = null
    this.forwardLayerIndex = 0
    this.backwardLayerIndex = null
    this.currentSnapshot = this.createIdleSnapshot()
    this.activations = cloneLayers(this.currentSnapshot.meta.forward)
  }

  createSnapshot({ input, target, phase, trace, loss, step, activeLayer }) {
    return {
      w: cloneWeights(this.weights),
      b: [],
      loss,
      step,
      epoch: step,
      phase,
      sampleIndex: 0,
      sampleCount: this.trainingData.length,
      inp: input.slice(),
      tgt: target.slice(),
      meta: {
        phase,
        activeLayer,
        input: input.slice(),
        target: target.slice(),
        output: trace.prediction.slice(),
        forward: cloneLayers(trace.activations),
        gradients: cloneLayers(trace.deltas),
        connectionUpdates:
          phase === 'backward'
            ? cloneWeights(trace.sampleUpdates)
            : cloneWeights(trace.zeroUpdates),
        epochConnectionUpdates:
          phase === 'backward'
            ? cloneWeights(trace.sampleUpdates)
            : cloneWeights(trace.zeroUpdates),
        zValues: cloneLayers(trace.zLayers),
        sampleLoss: trace.sampleLoss,
        averageLoss: loss,
        epochNumber: step,
        sampleIndex: 0,
        sampleCount: this.trainingData.length,
      },
    }
  }

  createIdleSnapshot() {
    const sample = this.trainingData[0]
    const trace = buildTrace(this.weights, sample.input, sample.target, this.layers)
    const maskedTrace = {
      ...trace,
      activations: createMaskedActivations(trace, 0, this.layers),
    }

    return this.createSnapshot({
      input: sample.input,
      target: sample.target,
      phase: 'idle',
      trace: maskedTrace,
      loss: trace.sampleLoss,
      step: this.currentEpoch,
      activeLayer: 0,
    })
  }

  canStepForward() {
    return this.currentPhase !== 'backward'
  }

  canStepBackward() {
    return this.currentPhase === 'backward'
  }

  getPhase() {
    return this.currentPhase
  }

  getForwardLayerIndex() {
    return this.forwardLayerIndex
  }

  getBackwardLayerIndex() {
    return this.backwardLayerIndex
  }

  runForward() {
    if (!this.canStepForward()) {
      return this.currentSnapshot
    }

    const sample = this.trainingData[0]
    const trace = this.pendingTrace ?? buildTrace(this.weights, sample.input, sample.target, this.layers)
    const nextLayerIndex = Math.min(this.forwardLayerIndex + 1, this.layers.length - 1)

    this.pendingTrace = trace
    this.lastInput = sample.input.slice()
    this.forwardLayerIndex = nextLayerIndex
    this.activations = createMaskedActivations(trace, nextLayerIndex, this.layers)
    this.currentPhase = nextLayerIndex === this.layers.length - 1 ? 'backward' : 'forward'

    if (this.currentPhase === 'backward') {
      this.backwardLayerIndex = this.layers.length - 1
    }

    this.currentSnapshot = this.createSnapshot({
      input: sample.input,
      target: sample.target,
      phase: 'forward',
      trace,
      loss: trace.sampleLoss,
      step: this.currentEpoch,
      activeLayer: nextLayerIndex,
    })

    return this.currentSnapshot
  }

  runBackward() {
    if (!this.canStepBackward() || !this.pendingTrace) {
      return this.currentSnapshot
    }

    const sample = this.trainingData[0]
    const backwardTrace = this.pendingTrace
    const activeLayer = this.backwardLayerIndex ?? this.layers.length - 1

    this.lastInput = sample.input.slice()
    this.activations = cloneLayers(backwardTrace.activations)

    if (activeLayer > 1) {
      this.backwardLayerIndex = activeLayer - 1
      this.currentSnapshot = this.createSnapshot({
        input: sample.input,
        target: sample.target,
        phase: 'backward',
        trace: backwardTrace,
        loss: backwardTrace.sampleLoss,
        step: this.currentEpoch,
        activeLayer,
      })

      return this.currentSnapshot
    }

    this.weights = subtractScaledMatrix(this.weights, backwardTrace.sampleUpdates, this.learningRate)
    this.currentEpoch += 1

    const updatedTrace = buildTrace(this.weights, sample.input, sample.target, this.layers)
    this.lastInput = sample.input.slice()
    this.activations = cloneLayers(updatedTrace.activations)
    this.generatedEpochs.push({ averageLoss: updatedTrace.sampleLoss })

    if (this.generatedEpochs.length > this.maxHistory) {
      this.generatedEpochs = this.generatedEpochs.slice(-this.maxHistory)
    }

    this.currentPhase = 'ready'
    this.pendingTrace = null
    this.forwardLayerIndex = 0
    this.backwardLayerIndex = null
    this.currentSnapshot = this.createSnapshot({
      input: sample.input,
      target: sample.target,
      phase: 'backward',
      trace: backwardTrace,
      loss: updatedTrace.sampleLoss,
      step: this.currentEpoch,
      activeLayer,
    })

    return this.currentSnapshot
  }

  forwardInput(input) {
    this.lastInput = input.slice()
    this.activations[0] = input.slice()

    for (let layerIndex = 1; layerIndex < this.layers.length; layerIndex += 1) {
      const previousActivations = this.activations[layerIndex - 1]
      const sums = multiplyVectorMatrix(previousActivations, this.weights[layerIndex - 1])
      this.activations[layerIndex] = sigmoidVector(sums)
    }

    return this.activations
  }

  totalLoss() {
    const restoreInput = this.lastInput.slice()
    let loss = 0

    this.trainingData.forEach(({ input, target }) => {
      this.forwardInput(input)
      const output = this.activations[this.layers.length - 1]
      loss +=
        output.reduce((sum, value, index) => sum + Math.pow(target[index] - value, 2), 0) /
        output.length
    })

    this.forwardInput(restoreInput)
    return loss / this.trainingData.length
  }

  getCurrentSnapshot() {
    return this.currentSnapshot
  }

  getIncomingWeights(layerIndex, neuronIndex) {
    if (layerIndex === 0) {
      return []
    }

    return Array.from({ length: this.layers[layerIndex - 1] }, (_, sourceIndex) => {
      return this.weights[layerIndex - 1][sourceIndex][neuronIndex]
    })
  }

  getDatasetPredictions() {
    const restoreInput = this.lastInput.slice()

    const predictions = this.trainingData.map(({ input, target }) => {
      this.forwardInput(input)

      return {
        input: input.slice(),
        target: target.slice(),
        prediction: this.activations[this.layers.length - 1].slice(),
      }
    })

    this.forwardInput(restoreInput)
    return predictions
  }

  getXorPredictions() {
    return this.getDatasetPredictions()
  }

  predict(input) {
    const restoreInput = this.lastInput.slice()

    this.forwardInput(input)
    const output = this.activations[this.layers.length - 1].slice()
    this.forwardInput(restoreInput)

    return output
  }

  getDecisionGrid(resolution) {
    if (this.layers[0] !== 2 || this.layers[this.layers.length - 1] !== 1) {
      return null
    }

    const restoreInput = this.lastInput.slice()
    const safeResolution = Math.max(4, resolution)

    const grid = Array.from({ length: safeResolution }, (_, rowIndex) => {
      const y = 1 - rowIndex / (safeResolution - 1)

      return Array.from({ length: safeResolution }, (_, columnIndex) => {
        const x = columnIndex / (safeResolution - 1)

        this.forwardInput([x, y])
        return this.activations[this.layers.length - 1][0]
      })
    })

    this.forwardInput(restoreInput)
    return grid
  }
}
