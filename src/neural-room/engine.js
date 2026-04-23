import {
  EXPECTED_EPOCH_LOSSES,
  INITIAL_WEIGHTS,
  LAYERS,
  LEARNING_RATE,
  MAX_HISTORY,
  TRAINING_DATA,
} from './constants.js'

function cloneWeights(weights) {
  return weights.map((layerWeights) =>
    layerWeights.map((row) => row.slice()),
  )
}

function cloneLayers(values) {
  return values.map((layerValues) => layerValues.slice())
}

function createZeroWeightUpdates(weights) {
  return weights.map((layerWeights) =>
    layerWeights.map((row) => row.map(() => 0)),
  )
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

function addMatrixInPlace(target, source) {
  for (let rowIndex = 0; rowIndex < target.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < target[rowIndex].length; columnIndex += 1) {
      target[rowIndex][columnIndex] += source[rowIndex][columnIndex]
    }
  }
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

function createEpochTrace(weights) {
  const batchUpdates = createZeroWeightUpdates(weights)
  const sampleTraces = TRAINING_DATA.map(({ input, target }) => {
    const z1 = multiplyVectorMatrix(input, weights[0])
    const a1 = sigmoidVector(z1)
    const z2 = multiplyVectorMatrix(a1, weights[1])
    const a2 = sigmoidVector(z2)
    const z3 = multiplyVectorMatrix(a2, weights[2])
    const a3 = sigmoidVector(z3)
    const prediction = a3[0]
    const expected = target[0]
    const sampleLoss = Math.pow(expected - prediction, 2)
    const d3 = [-2 * (expected - prediction) * prediction * (1 - prediction)]
    const d2 = a2.map(
      (activation, nodeIndex) =>
        d3[0] * weights[2][nodeIndex][0] * activation * (1 - activation),
    )
    const d1 = a1.map((activation, nodeIndex) => {
      let error = 0

      for (let nextNodeIndex = 0; nextNodeIndex < d2.length; nextNodeIndex += 1) {
        error += d2[nextNodeIndex] * weights[1][nodeIndex][nextNodeIndex]
      }

      return error * activation * (1 - activation)
    })

    const sampleUpdates = [
      outerProduct(input, d1),
      outerProduct(a1, d2),
      outerProduct(a2, d3),
    ]

    addMatrixInPlace(batchUpdates[0], sampleUpdates[0])
    addMatrixInPlace(batchUpdates[1], sampleUpdates[1])
    addMatrixInPlace(batchUpdates[2], sampleUpdates[2])

    return {
      input: input.slice(),
      target: target.slice(),
      prediction,
      sampleLoss,
      zLayers: [input.slice(), z1, z2, z3],
      activations: [input.slice(), a1, a2, a3],
      deltas: [Array.from({ length: LAYERS[0] }, () => 0), d1, d2, d3],
      sampleUpdates,
    }
  })

  const averageLoss =
    sampleTraces.reduce((sum, trace) => sum + trace.sampleLoss, 0) / sampleTraces.length

  return {
    weights: cloneWeights(weights),
    averageLoss,
    batchUpdates,
    sampleTraces,
    nextWeights: subtractScaledMatrix(weights, batchUpdates, LEARNING_RATE),
  }
}

export class NeuralNetworkEngine {
  constructor() {
    this.layers = LAYERS
    this.learningRate = LEARNING_RATE
    this.maxHistory = MAX_HISTORY
    this.history = []
    this.historyIndex = -1
    this.generatedEpochs = []
    this.activations = this.layers.map((count) => Array.from({ length: count }, () => 0))
    this.biases = []
    this.weights = cloneWeights(INITIAL_WEIGHTS)
    this.lastInput = TRAINING_DATA[0].input.slice()

    this.initialize()
  }

  initialize() {
    this.history = []
    this.historyIndex = -1
    this.generatedEpochs = []
    this.weights = cloneWeights(INITIAL_WEIGHTS)
    this.lastInput = TRAINING_DATA[0].input.slice()
    this.activations = this.layers.map((count) => Array.from({ length: count }, () => 0))

    this.ensureFrame(0)
    this.applySnapshot(this.getCurrentSnapshot())
  }

  ensureFrame(frameIndex) {
    while (this.history.length <= frameIndex) {
      const baseWeights =
        this.generatedEpochs.length === 0
          ? cloneWeights(INITIAL_WEIGHTS)
          : cloneWeights(this.generatedEpochs[this.generatedEpochs.length - 1].nextWeights)

      const epochTrace = createEpochTrace(baseWeights)
      const epochNumber = this.generatedEpochs.length + 1
      this.generatedEpochs.push(epochTrace)

      epochTrace.sampleTraces.forEach((sampleTrace, sampleIndex) => {
        this.history.push({
          w: cloneWeights(epochTrace.weights),
          b: [],
          loss: epochTrace.averageLoss,
          step: epochNumber,
          epoch: epochNumber,
          sampleIndex,
          sampleCount: epochTrace.sampleTraces.length,
          inp: sampleTrace.input.slice(),
          tgt: sampleTrace.target.slice(),
          meta: this.buildSnapshotMeta(epochTrace, sampleTrace, epochNumber, sampleIndex),
        })
      })
    }

    if (this.historyIndex === -1 && this.history.length > 0) {
      this.historyIndex = 0
    }

    return this.history[frameIndex] ?? this.history[this.history.length - 1] ?? null
  }

  buildSnapshotMeta(epochTrace, sampleTrace, epochNumber, sampleIndex) {
    return {
      input: sampleTrace.input.slice(),
      target: sampleTrace.target.slice(),
      output: [sampleTrace.prediction],
      forward: cloneLayers(sampleTrace.activations),
      gradients: cloneLayers(sampleTrace.deltas),
      connectionUpdates: cloneWeights(sampleTrace.sampleUpdates),
      epochConnectionUpdates: cloneWeights(epochTrace.batchUpdates),
      zValues: cloneLayers(sampleTrace.zLayers),
      sampleLoss: sampleTrace.sampleLoss,
      averageLoss: epochTrace.averageLoss,
      epochNumber,
      sampleIndex,
      sampleCount: epochTrace.sampleTraces.length,
      expectedLoss:
        EXPECTED_EPOCH_LOSSES[epochNumber - 1] ?? epochTrace.averageLoss,
    }
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

    TRAINING_DATA.forEach(({ input, target }) => {
      this.forwardInput(input)
      loss += Math.pow(target[0] - this.activations[this.layers.length - 1][0], 2)
    })

    this.forwardInput(restoreInput)
    return loss / TRAINING_DATA.length
  }

  applySnapshot(snapshot) {
    if (!snapshot) {
      return
    }

    this.weights = cloneWeights(snapshot.w)
    this.biases = []
    this.forwardInput(snapshot.inp)
  }

  getCurrentSnapshot() {
    return this.history[this.historyIndex] ?? null
  }

  stepForward() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex += 1
      this.applySnapshot(this.getCurrentSnapshot())
      return this.getCurrentSnapshot()
    }

    const nextFrame = this.ensureFrame(this.history.length)

    if (!nextFrame || this.historyIndex >= this.history.length - 1) {
      this.applySnapshot(this.getCurrentSnapshot())
      return this.getCurrentSnapshot()
    }

    this.historyIndex += 1
    this.applySnapshot(this.getCurrentSnapshot())
    return this.getCurrentSnapshot()
  }

  stepBack() {
    if (this.historyIndex <= 0) {
      this.applySnapshot(this.getCurrentSnapshot())
      return this.getCurrentSnapshot()
    }

    this.historyIndex -= 1
    this.applySnapshot(this.getCurrentSnapshot())
    return this.getCurrentSnapshot()
  }

  stepForwardNav() {
    return this.stepForward()
  }

  stepEpochForward() {
    const currentSnapshot = this.getCurrentSnapshot()

    if (!currentSnapshot) {
      return this.ensureFrame(0)
    }

    const targetIndex = this.historyIndex + (currentSnapshot.sampleCount - currentSnapshot.sampleIndex)
    this.ensureFrame(targetIndex)
    this.historyIndex = Math.min(targetIndex, this.history.length - 1)
    this.applySnapshot(this.getCurrentSnapshot())

    return this.getCurrentSnapshot()
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

    const predictions = TRAINING_DATA.map(({ input, target }) => {
      this.forwardInput(input)

      return {
        input: input.slice(),
        target: target.slice(),
        prediction: this.activations[this.layers.length - 1][0],
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
    const output = this.activations[this.layers.length - 1][0]
    this.forwardInput(restoreInput)

    return output
  }

  getDecisionGrid(resolution) {
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
