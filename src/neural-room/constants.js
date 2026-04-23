import {
  EXCEL_EXPECTED_EPOCH_LOSSES,
  EXCEL_INITIAL_WEIGHTS,
  EXCEL_TRAINING_DATA,
} from './excel-data.js'

export const COLORS = {
  background: 0x04070b,
  roomBack: 0x1d313c,
  roomSide: 0x1a2832,
  roomFloor: 0x142631,
  roomCeiling: 0x1a2a34,
  roomPanel: 0x27414e,
  roomTrim: 0x4e8099,
  roomAccent: 0x67d7ff,
  roomGlass: 0x9fd9e6,
  roomWarning: 0xffbf73,
  grid: 0x13293a,
  ambient: 0xb1c7da,
  directional: 0xf0f8ff,
  point: 0x72c7ff,
  positive: 0x58b7ff,
  negative: 0xff6e7d,
  neutral: 0x79a2b4,
  selected: 0x9de7ff,
  nodeLight: 0x3fa2d8,
  forward: 0x8ff4ff,
  backward: 0xffa45b,
  hologramLow: 0x102a36,
  hologramMid: 0x0f8ca3,
  hologramHigh: 0x93f7ff,
  hologramEdge: 0xffd38a,
}

export const LAYERS = [2, 3, 2, 1]
export const LAYER_NAMES = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
export const LAYER_FUNCTIONS = ['Linear', 'Sigmoid', 'Sigmoid', 'Sigmoid']
export const TRAINING_DATA = EXCEL_TRAINING_DATA
export const INITIAL_WEIGHTS = EXCEL_INITIAL_WEIGHTS
export const EXPECTED_EPOCH_LOSSES = EXCEL_EXPECTED_EPOCH_LOSSES
export const TOTAL_EPOCHS = EXPECTED_EPOCH_LOSSES.length

export const LEARNING_RATE = 0.0025
export const MAX_HISTORY = 500
export const ARCHITECTURE_LABEL = '2 · 3 · 2 · 1'

export const LAYER_SPACING = 7.1
export const NODE_SPACING = 3.7

export const ROOM_DIMENSIONS = {
  width: 44,
  height: 24,
  depth: 44,
}

export const CAMERA_DEFAULT = {
  x: -11.5,
  y: 2.6,
  z: 5,
}

export const CAMERA_ENTRY = {
  x: 0,
  y: 2,
  z: 17,
}

export const MOVE_SPEED = 0.13
export const DASH_COUNT = 4
export const DASH_GAP = 1 / DASH_COUNT
export const DECISION_SURFACE_RESOLUTION = 56

export const TOTAL_CONNECTIONS = LAYERS.slice(0, -1).reduce(
  (sum, count, index) => sum + count * LAYERS[index + 1],
  0,
)
