function createRandomWeights(layers) {
  return layers.slice(0, -1).map((fromCount, layerIndex) =>
    Array.from({ length: fromCount }, () =>
      Array.from({ length: layers[layerIndex + 1] }, () => Number((Math.random() * 2 - 1).toFixed(3))),
    ),
  )
}

function createLayerNames(layers) {
  return layers.map((_, index) => {
    if (index === 0) {
      return 'Input'
    }

    if (index === layers.length - 1) {
      return 'Output'
    }

    return `Hidden ${index}`
  })
}

function createLayerFunctions(layers) {
  return layers.map((_, index) => (index === 0 ? 'Linear' : 'Sigmoid'))
}

function countConnections(layers) {
  return layers.slice(0, -1).reduce((sum, count, index) => sum + count * layers[index + 1], 0)
}

export function buildModelConfig({ layers, input, target }) {
  if (!Array.isArray(layers) || layers.length < 2) {
    throw new Error('La arquitectura debe tener al menos capa de entrada y salida.')
  }

  if (input.length !== layers[0]) {
    throw new Error('El vector de entrada no coincide con la capa de entrada.')
  }

  if (target.length !== layers[layers.length - 1]) {
    throw new Error('El vector objetivo no coincide con la capa de salida.')
  }

  const normalizedLayers = layers.map((value) => Number(value))
  const normalizedInput = input.map((value) => Number(value))
  const normalizedTarget = target.map((value) => Number(value))

  return {
    layers: normalizedLayers,
    layerNames: createLayerNames(normalizedLayers),
    layerFunctions: createLayerFunctions(normalizedLayers),
    architectureLabel: normalizedLayers.join(' · '),
    connectionCount: countConnections(normalizedLayers),
    trainingData: [
      {
        input: normalizedInput,
        target: normalizedTarget,
      },
    ],
    initialWeights: createRandomWeights(normalizedLayers),
  }
}

export function createDefaultModelConfig() {
  return buildModelConfig({
    layers: [2, 3, 2, 1],
    input: [0.7, 0.6],
    target: [0.1],
  })
}
