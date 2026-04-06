import {
  LAYERS,
  LEARNING_RATE,
  MAX_HISTORY,
  XOR_DATA,
} from './constants.js'

function cloneWeights(weights) {
  return weights.map((layerWeights) =>
    layerWeights.map((row) => row.slice()),
  )
}

function cloneBiases(biases) {
  return biases.map((layerBiases) => layerBiases.slice())
}

function cloneLayers(values) {
  return values.map((layerValues) => layerValues.slice())
}

function createZeroWeightUpdates(weights) {
  return weights.map((layerWeights) =>
    layerWeights.map((row) => row.map(() => 0)),
  )
}

export class NeuralNetworkEngine {
  constructor() {
    this.layers = LAYERS
    this.learningRate = LEARNING_RATE
    this.maxHistory = MAX_HISTORY
    this.history = []
    this.historyIndex = -1
    this.dataIndex = 0
    this.lastInput = [0, 0]
    this.weights = []
    this.biases = []
    this.activations = this.layers.map((count) => Array.from({ length: count }, () => 0))

    this.initialize()
  }

  initialize() {
    this.weights = []

    for (let layerIndex = 0; layerIndex < this.layers.length - 1; layerIndex += 1) {
      const layerWeights = []

      for (let fromIndex = 0; fromIndex < this.layers[layerIndex]; fromIndex += 1) {
        const row = []

        for (let toIndex = 0; toIndex < this.layers[layerIndex + 1]; toIndex += 1) {
          row.push(this.randomWeight())
        }

        layerWeights.push(row)
      }

      this.weights.push(layerWeights)
    }

    this.biases = []

    for (let layerIndex = 0; layerIndex < this.layers.length - 1; layerIndex += 1) {
      this.biases.push(
        Array.from(
          { length: this.layers[layerIndex + 1] },
          () => this.randomWeight() * 0.1,
        ),
      )
    }

    this.activations = this.layers.map((count) =>
      Array.from({ length: count }, () => 0),
    )
    this.history = []
    this.historyIndex = -1
    this.dataIndex = 0
    this.lastInput = [0, 0]

    this.forwardInput([0, 0])
    this.saveSnapshot(this.totalLoss(), [0, 0], [0], this.buildSnapshotMeta([0, 0], [0]))
    this.applySnapshot(this.getCurrentSnapshot())
  }

  buildSnapshotMeta(input, target, gradients, connectionUpdates, activations) {
    const zeroGradients = this.layers.map((count) =>
      Array.from({ length: count }, () => 0),
    )

    return {
      input: input.slice(),
      target: target.slice(),
      output: (activations ?? this.activations)[this.layers.length - 1].slice(),
      forward: cloneLayers(activations ?? this.activations),
      gradients: cloneLayers(gradients ?? zeroGradients),
      connectionUpdates: cloneWeights(connectionUpdates ?? createZeroWeightUpdates(this.weights)),
    }
  }

  randomWeight() {
    return (Math.random() * 2 - 1) * 0.8
  }

  sigmoid(value) {
    return 1 / (1 + Math.exp(-value))
  }

  relu(value) {
    return Math.max(0, value)
  }

  reluDerivative(value) {
    return value > 0 ? 1 : 0
  }

  forwardInput(input) {
    this.lastInput = input.slice()
    this.activations[0][0] = input[0]
    this.activations[0][1] = input[1]

    for (let layerIndex = 1; layerIndex < this.layers.length; layerIndex += 1) {
      for (let nodeIndex = 0; nodeIndex < this.layers[layerIndex]; nodeIndex += 1) {
        let sum = this.biases[layerIndex - 1][nodeIndex]

        for (
          let sourceIndex = 0;
          sourceIndex < this.layers[layerIndex - 1];
          sourceIndex += 1
        ) {
          sum +=
            this.activations[layerIndex - 1][sourceIndex] *
            this.weights[layerIndex - 1][sourceIndex][nodeIndex]
        }

        this.activations[layerIndex][nodeIndex] =
          layerIndex === this.layers.length - 1 ? this.sigmoid(sum) : this.relu(sum)
      }
    }

    return this.activations
  }

  backprop(input, target) {
    this.forwardInput(input)
    const forwardActivations = cloneLayers(this.activations)

    const deltas = this.layers.map((count) =>
      Array.from({ length: count }, () => 0),
    )
    const lastLayerIndex = this.layers.length - 1
    const connectionUpdates = createZeroWeightUpdates(this.weights)

    for (let nodeIndex = 0; nodeIndex < this.layers[lastLayerIndex]; nodeIndex += 1) {
      const output = this.activations[lastLayerIndex][nodeIndex]
      deltas[lastLayerIndex][nodeIndex] =
        (output - target[nodeIndex]) * output * (1 - output)
    }

    for (let layerIndex = lastLayerIndex - 1; layerIndex >= 1; layerIndex -= 1) {
      for (let nodeIndex = 0; nodeIndex < this.layers[layerIndex]; nodeIndex += 1) {
        let error = 0

        for (
          let nextNodeIndex = 0;
          nextNodeIndex < this.layers[layerIndex + 1];
          nextNodeIndex += 1
        ) {
          error +=
            deltas[layerIndex + 1][nextNodeIndex] *
            this.weights[layerIndex][nodeIndex][nextNodeIndex]
        }

        deltas[layerIndex][nodeIndex] =
          error * this.reluDerivative(this.activations[layerIndex][nodeIndex])
      }
    }

    for (let layerIndex = 0; layerIndex < this.layers.length - 1; layerIndex += 1) {
      for (let nodeIndex = 0; nodeIndex < this.layers[layerIndex + 1]; nodeIndex += 1) {
        this.biases[layerIndex][nodeIndex] -=
          this.learningRate * deltas[layerIndex + 1][nodeIndex]

        for (
          let sourceIndex = 0;
          sourceIndex < this.layers[layerIndex];
          sourceIndex += 1
        ) {
          const weightShift =
            this.learningRate *
            deltas[layerIndex + 1][nodeIndex] *
            this.activations[layerIndex][sourceIndex]

          connectionUpdates[layerIndex][sourceIndex][nodeIndex] = weightShift
          this.weights[layerIndex][sourceIndex][nodeIndex] -=
            weightShift
        }
      }
    }

    let loss = 0

    for (let nodeIndex = 0; nodeIndex < this.layers[lastLayerIndex]; nodeIndex += 1) {
      loss +=
        0.5 *
        Math.pow(this.activations[lastLayerIndex][nodeIndex] - target[nodeIndex], 2)
    }

    return {
      loss,
      meta: this.buildSnapshotMeta(
        input,
        target,
        deltas,
        connectionUpdates,
        forwardActivations,
      ),
    }
  }

  totalLoss() {
    const restoreInput = this.lastInput.slice()
    let loss = 0

    for (let pairIndex = 0; pairIndex < XOR_DATA.length; pairIndex += 1) {
      this.forwardInput(XOR_DATA[pairIndex][0])
      const output = this.activations[this.layers.length - 1][0]
      loss += 0.5 * Math.pow(output - XOR_DATA[pairIndex][1][0], 2)
    }

    this.forwardInput(restoreInput)
    return loss
  }

  saveSnapshot(loss, input, target, meta = this.buildSnapshotMeta(input, target)) {
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1)
    }

    this.history.push({
      w: cloneWeights(this.weights),
      b: cloneBiases(this.biases),
      loss,
      step: this.history.length,
      inp: input.slice(),
      tgt: target.slice(),
      meta,
    })

    if (this.history.length > this.maxHistory) {
      this.history.shift()
    }

    this.historyIndex = this.history.length - 1
  }

  applySnapshot(snapshot) {
    this.weights = cloneWeights(snapshot.w)
    this.biases = cloneBiases(snapshot.b)
    this.forwardInput(snapshot.inp)
  }

  getCurrentSnapshot() {
    return this.history[this.historyIndex]
  }

  stepForward() {
    const pair = XOR_DATA[this.dataIndex % XOR_DATA.length]
    this.dataIndex += 1
    const result = this.backprop(pair[0], pair[1])
    this.saveSnapshot(this.totalLoss(), pair[0], pair[1], result.meta)
    this.applySnapshot(this.getCurrentSnapshot())
    return this.getCurrentSnapshot()
  }

  stepBack() {
    if (this.historyIndex <= 0) {
      return this.getCurrentSnapshot()
    }

    this.historyIndex -= 1
    this.applySnapshot(this.getCurrentSnapshot())
    return this.getCurrentSnapshot()
  }

  stepForwardNav() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex += 1
      this.applySnapshot(this.getCurrentSnapshot())
      return this.getCurrentSnapshot()
    }

    return this.stepForward()
  }

  getIncomingWeights(layerIndex, neuronIndex) {
    if (layerIndex === 0) {
      return []
    }

    return Array.from({ length: this.layers[layerIndex - 1] }, (_, sourceIndex) => {
      return this.weights[layerIndex - 1][sourceIndex][neuronIndex]
    })
  }

  getXorPredictions() {
    const restoreInput = this.lastInput.slice()

    const predictions = XOR_DATA.map(([input, target]) => {
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
