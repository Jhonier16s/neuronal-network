import { LEARNING_RATE } from './constants.js'

function formatNumber(value, digits = 2) {
  return String(Number(value.toFixed(digits)))
}

function formatVector(values, digits = 2) {
  return `[${values.map((value) => formatNumber(value, digits)).join(', ')}]`
}

function formatSampleLabel(input, target) {
  return `${formatVector(input)} -> ${formatVector(target)}`
}

function createPlaceholderOutputs(trainingData) {
  return trainingData.map(({ input, target }) => ({
    label: formatSampleLabel(input, target),
    value: '—',
    correct: null,
  }))
}

export function createInitialViewState(modelConfig) {
  return {
    architectureLabel: modelConfig.architectureLabel,
    connectionCount: modelConfig.connectionCount,
    epoch: 0,
    entryVisible: true,
    mode3D: true,
    pointerLocked: false,
    cinematicActive: false,
    mouseLabel: 'Clic para explorar · WASD · ESC',
    infoVisible: false,
    selectedNeuron: null,
    training: {
      step: 0,
      lossText: '—',
      lossTone: '',
      lrText: String(LEARNING_RATE),
      statusText: 'Listo',
      statusTone: '',
      stepPosition: '0 / 0',
      sampleText: '—',
      phaseText: 'Esperando forward',
      visualText: 'Forward + backprop + visual 3D',
      outputs: createPlaceholderOutputs(modelConfig.trainingData),
      losses: [0],
      currentLossIndex: 0,
      canForward: true,
      canBackward: false,
      autoActive: false,
      autoSpeed: 1.5,
    },
  }
}
