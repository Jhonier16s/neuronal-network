export const COLORS = {
  background: 0x06090d,
  roomBack: 0x060d14,
  roomSide: 0x060b12,
  roomFloor: 0x07101a,
  roomCeiling: 0x050810,
  grid: 0x0d2030,
  ambient: 0x0d2038,
  directional: 0x4a80d0,
  point: 0x2a5898,
  positive: 0x4a90d9,
  negative: 0xd05070,
  neutral: 0x507888,
  selected: 0x80c0ff,
  nodeLight: 0x3a80c0,
}

export const LAYERS = [2, 3, 2, 1]
export const LAYER_NAMES = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
export const LAYER_FUNCTIONS = ['Linear', 'ReLU', 'ReLU', 'Sigmoid']
export const XOR_DATA = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]],
]

export const LEARNING_RATE = 0.5
export const MAX_HISTORY = 500
export const ARCHITECTURE_LABEL = '2 · 3 · 2 · 1'

export const LAYER_SPACING = 6.5
export const NODE_SPACING = 3.4

export const ROOM_DIMENSIONS = {
  width: 70,
  height: 24,
  depth: 32,
}

export const CAMERA_DEFAULT = {
  x: -14,
  y: 1.6,
  z: 0,
}

export const CAMERA_ENTRY = {
  x: 0,
  y: 1.6,
  z: 22,
}

export const MOVE_SPEED = 0.13
export const DASH_COUNT = 4
export const DASH_GAP = 1 / DASH_COUNT

export const TOTAL_CONNECTIONS = LAYERS.slice(0, -1).reduce(
  (sum, count, index) => sum + count * LAYERS[index + 1],
  0,
)
