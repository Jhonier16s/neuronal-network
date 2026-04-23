import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import {
  ARCHITECTURE_LABEL,
  CAMERA_DEFAULT,
  CAMERA_ENTRY,
  COLORS,
  DASH_COUNT,
  DASH_GAP,
  DECISION_SURFACE_RESOLUTION,
  LAYERS,
  LAYER_FUNCTIONS,
  LAYER_NAMES,
  LAYER_SPACING,
  LEARNING_RATE,
  MOVE_SPEED,
  NODE_SPACING,
  ROOM_DIMENSIONS,
  TOTAL_CONNECTIONS,
  TRAINING_DATA,
} from './constants.js'
import { NeuralNetworkEngine } from './engine.js'

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function lerp(start, end, alpha) {
  return start + (end - start) * alpha
}

function smoothstep01(value) {
  const clamped = clamp(value, 0, 1)
  return clamped * clamped * (3 - 2 * clamped)
}

function getWeightColor3(weight) {
  if (weight > 0.1) {
    return new THREE.Color(COLORS.positive)
  }

  if (weight < -0.1) {
    return new THREE.Color(COLORS.negative)
  }

  return new THREE.Color(COLORS.neutral)
}

function getWeightColorCss(weight) {
  if (weight > 0.1) {
    return '#4a90d9'
  }

  if (weight < -0.1) {
    return '#d05070'
  }

  return '#507888'
}

function getActivationTone(value) {
  if (value > 0.5) {
    return 'p'
  }

  if (value < 0.15) {
    return 'n'
  }

  return 'm'
}

function formatSampleLabel(input, target) {
  return `${input[0]},${input[1]} → ${target[0]}`
}

function isPredictionAligned(prediction, target) {
  return Math.abs(prediction - target[0]) < 0.05
}

function createPlaceholderOutputs() {
  return TRAINING_DATA.map(({ input, target }) => ({
    label: formatSampleLabel(input, target),
    value: '—',
    correct: null,
  }))
}

function getStatusMeta(statusMode) {
  switch (statusMode) {
    case 'auto':
      return { text: 'Auto ▶', tone: 'ok' }
    case 'paused':
      return { text: 'Pausado', tone: '' }
    case 'converged':
      return { text: 'Modelo Excel listo', tone: 'good' }
    default:
      return { text: 'Listo', tone: '' }
  }
}

function buildWeightTexture(weight) {
  const canvas = document.createElement('canvas')
  const text = `${weight >= 0 ? '+' : ''}${weight.toFixed(3)}`
  const color = getWeightColorCss(weight)

  canvas.width = 180
  canvas.height = 52

  const context = canvas.getContext('2d')

  context.fillStyle = 'rgba(3, 6, 10, 0.88)'
  context.fillRect(0, 0, 180, 52)
  context.strokeStyle = `${color}66`
  context.lineWidth = 1
  context.strokeRect(0.5, 0.5, 179, 51)
  context.font = '500 15px IBM Plex Mono, monospace'
  context.fillStyle = color
  context.textAlign = 'center'
  context.shadowColor = color
  context.shadowBlur = 6
  context.fillText(text, 90, 33)

  return new THREE.CanvasTexture(canvas)
}

function buildLayerTexture(name, activationFn) {
  const canvas = document.createElement('canvas')

  canvas.width = 320
  canvas.height = 80

  const context = canvas.getContext('2d')

  context.font = '500 20px IBM Plex Mono, monospace'
  context.fillStyle = '#6ab0f0'
  context.textAlign = 'center'
  context.shadowColor = '#3a80d0'
  context.shadowBlur = 5
  context.fillText(name.toUpperCase(), 160, 28)
  context.font = '300 13px IBM Plex Mono, monospace'
  context.fillStyle = '#4a6878'
  context.shadowBlur = 0
  context.fillText(activationFn, 160, 52)

  return new THREE.CanvasTexture(canvas)
}

function buildMiniLabelTexture(text) {
  const canvas = document.createElement('canvas')

  canvas.width = 150
  canvas.height = 48

  const context = canvas.getContext('2d')

  context.fillStyle = 'rgba(4, 8, 14, 0.82)'
  context.fillRect(0, 0, canvas.width, canvas.height)
  context.strokeStyle = '#225674'
  context.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1)
  context.font = '500 16px IBM Plex Mono, monospace'
  context.fillStyle = '#8ef0ff'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(text.toUpperCase(), canvas.width / 2, canvas.height / 2)

  return new THREE.CanvasTexture(canvas)
}

function buildResearchPanelTexture(title, subtitle, lines, accent = '#7bd5ff') {
  const canvas = document.createElement('canvas')

  canvas.width = 512
  canvas.height = 280

  const context = canvas.getContext('2d')

  context.fillStyle = 'rgba(5, 10, 16, 0.95)'
  context.fillRect(0, 0, canvas.width, canvas.height)

  const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height)
  gradient.addColorStop(0, 'rgba(20, 58, 86, 0.32)')
  gradient.addColorStop(1, 'rgba(6, 12, 18, 0)')
  context.fillStyle = gradient
  context.fillRect(0, 0, canvas.width, canvas.height)

  context.strokeStyle = `${accent}70`
  context.lineWidth = 2
  context.strokeRect(8, 8, canvas.width - 16, canvas.height - 16)

  context.fillStyle = accent
  context.font = '600 28px IBM Plex Mono, monospace'
  context.textAlign = 'left'
  context.fillText(title.toUpperCase(), 32, 54)

  context.font = '500 16px IBM Plex Mono, monospace'
  context.fillStyle = '#8fb8d4'
  context.fillText(subtitle.toUpperCase(), 32, 84)

  context.strokeStyle = `${accent}45`
  context.beginPath()
  context.moveTo(32, 104)
  context.lineTo(canvas.width - 32, 104)
  context.stroke()

  context.font = '400 16px IBM Plex Mono, monospace'
  context.fillStyle = '#d2deea'
  lines.forEach((line, index) => {
    context.fillText(line, 32, 144 + index * 32)
  })

  context.fillStyle = 'rgba(123, 213, 255, 0.3)'
  for (let offset = 0; offset < 5; offset += 1) {
    context.fillRect(32, 214 + offset * 10, 190 - offset * 18, 2)
  }

  return new THREE.CanvasTexture(canvas)
}

function buildCreditsTexture(names, accent = '#8ef0ff') {
  const canvas = document.createElement('canvas')

  canvas.width = 720
  canvas.height = 300

  const context = canvas.getContext('2d')

  context.fillStyle = 'rgba(7, 14, 20, 0.96)'
  context.fillRect(0, 0, canvas.width, canvas.height)

  const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height)
  gradient.addColorStop(0, 'rgba(26, 70, 96, 0.24)')
  gradient.addColorStop(1, 'rgba(8, 16, 24, 0)')
  context.fillStyle = gradient
  context.fillRect(0, 0, canvas.width, canvas.height)

  context.strokeStyle = `${accent}66`
  context.lineWidth = 2
  context.strokeRect(10, 10, canvas.width - 20, canvas.height - 20)

  context.fillStyle = '#9eb6c8'
  context.font = '600 18px Rajdhani, sans-serif'
  context.letterSpacing = '2px'
  context.fillText('INTEGRANTES', 44, 58)

  context.strokeStyle = `${accent}44`
  context.beginPath()
  context.moveTo(44, 82)
  context.lineTo(canvas.width - 44, 82)
  context.stroke()

  context.font = '700 42px Rajdhani, sans-serif'
  context.fillStyle = accent
  names.forEach((name, index) => {
    context.fillText(name.toUpperCase(), 44, 144 + index * 66)
  })

  context.font = '500 16px IBM Plex Mono, monospace'
  context.fillStyle = 'rgba(190, 214, 228, 0.76)'
  context.fillText('NEURAL ROOM · THREE.JS EXPERIENCE', 44, canvas.height - 38)

  return new THREE.CanvasTexture(canvas)
}

function buildDisplayPanelTexture(title, accent) {
  const canvas = document.createElement('canvas')

  canvas.width = 768
  canvas.height = 384

  const context = canvas.getContext('2d')
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace

  return { canvas, context, texture, title, accent }
}

function renderDisplayPanelTexture(panel, { tick, loss, epoch, outputs, activity }) {
  const { canvas, context, title, accent } = panel
  const width = canvas.width
  const height = canvas.height
  const accentColor = accent
  const dim = 'rgba(167, 205, 224, 0.72)'

  context.clearRect(0, 0, width, height)

  const bg = context.createLinearGradient(0, 0, width, height)
  bg.addColorStop(0, 'rgba(6, 13, 20, 0.98)')
  bg.addColorStop(1, 'rgba(10, 24, 34, 0.98)')
  context.fillStyle = bg
  context.fillRect(0, 0, width, height)

  const haze = context.createRadialGradient(width * 0.78, height * 0.2, 0, width * 0.78, height * 0.2, width * 0.5)
  haze.addColorStop(0, `${accentColor}33`)
  haze.addColorStop(1, 'rgba(0,0,0,0)')
  context.fillStyle = haze
  context.fillRect(0, 0, width, height)

  context.strokeStyle = `${accentColor}66`
  context.lineWidth = 3
  context.strokeRect(10, 10, width - 20, height - 20)

  context.fillStyle = accentColor
  context.font = '700 34px IBM Plex Mono, monospace'
  context.fillText(title.toUpperCase(), 34, 52)

  context.fillStyle = dim
  context.font = '500 16px IBM Plex Mono, monospace'
  context.fillText('LIVE SYSTEM VISUALIZATION', 36, 82)

  context.strokeStyle = `${accentColor}40`
  context.lineWidth = 2
  context.beginPath()
  context.moveTo(34, 102)
  context.lineTo(width - 34, 102)
  context.stroke()

  const spark = 0.35 + activity * 0.4
  const graphBaseY = 260

  context.beginPath()
  for (let index = 0; index < 52; index += 1) {
    const x = 34 + (index / 51) * (width - 68)
    const wave = Math.sin(tick * 1.8 + index * 0.22) * 16
    const wave2 = Math.cos(tick * 1.1 + index * 0.36) * 9
    const y = graphBaseY - spark * 30 - wave - wave2

    if (index === 0) {
      context.moveTo(x, y)
    } else {
      context.lineTo(x, y)
    }
  }
  context.strokeStyle = accentColor
  context.lineWidth = 3
  context.shadowColor = accentColor
  context.shadowBlur = 16
  context.stroke()
  context.shadowBlur = 0

  context.beginPath()
  for (let index = 0; index < 52; index += 1) {
    const x = 34 + (index / 51) * (width - 68)
    const wave = Math.sin(tick * 1.35 + index * 0.18 + 1.2) * 10
    const y = graphBaseY + 42 - wave - loss * 75

    if (index === 0) {
      context.moveTo(x, y)
    } else {
      context.lineTo(x, y)
    }
  }
  context.strokeStyle = 'rgba(255, 211, 138, 0.8)'
  context.lineWidth = 2
  context.stroke()

  context.fillStyle = dim
  context.font = '600 14px IBM Plex Mono, monospace'
  context.fillText(`EPOCH ${String(epoch).padStart(3, '0')}`, 36, 132)
  context.fillText(`LOSS ${loss.toFixed(4)}`, 36, 156)
  context.fillText(`ACTIVITY ${activity.toFixed(2)}`, 36, 180)

  outputs.slice(0, 4).forEach((output, index) => {
    const y = 132 + index * 34
    context.fillStyle = output.correct ? '#7af0b2' : '#ff9ba7'
    context.fillRect(width - 248, y - 15, 10, 10)
    context.fillStyle = dim
    context.fillText(`${output.label}  ${output.value}`, width - 226, y - 5)
  })

  context.globalAlpha = 0.08
  for (let y = 0; y < height; y += 6) {
    context.fillStyle = '#8ef0ff'
    context.fillRect(0, y, width, 1)
  }
  context.globalAlpha = 1

  panel.texture.needsUpdate = true
}

export function createInitialViewState() {
  return {
    architectureLabel: ARCHITECTURE_LABEL,
    connectionCount: TOTAL_CONNECTIONS,
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
      visualText: 'Forward batch + backprop batch + visual 3D',
      outputs: createPlaceholderOutputs(),
      losses: [0],
      currentLossIndex: 0,
    },
  }
}

export class NeuralRoomController {
  constructor({ onStateChange }) {
    this.onStateChange = onStateChange
    this.engine = new NeuralNetworkEngine()
    this.entryVisible = true
    this.mode3D = true
    this.pointerLocked = false
    this.statusMode = 'ready'
    this.autoTrain = false
    this.viewportEl = null
    this.scene = null
    this.camera = null
    this.renderer = null
    this.composer = null
    this.renderPass = null
    this.bloomPass = null
    this.environmentGroup = null
    this.networkGroup = null
    this.neurons = []
    this.neuronMap = []
    this.nodePointMap = []
    this.connections = []
    this.weightSprites = []
    this.layerSprites = []
    this.trainingBursts = []
    this.decisionSurface = null
    this.labAccentMaterials = []
    this.labAccentLights = []
    this.labFans = []
    this.labSteam = []
    this.labBots = []
    this.labCoreRings = []
    this.displayPanels = []
    this.heroSpotlight = null
    this.selectedNeuron = null
    this.keys = {}
    this.listeners = []
    this.animationFrameId = null
    this.tick = 0
    this.frameCount = 0
    this.lastFrameTime = 0
    this.phase2d = 0
    this.camX = CAMERA_DEFAULT.x
    this.camY = CAMERA_DEFAULT.y
    this.camZ = CAMERA_DEFAULT.z
    this.yaw = 0
    this.pitch = 0
    this.tourActive = false
    this.tourShotIndex = 0
    this.tourShotTime = 0
    this.tourLabel = ''
    this.tourShots = []
    this.tourAnchors = new Map()
    this.tourCameraPoint = new THREE.Vector3()
    this.tourLookPoint = new THREE.Vector3()
    this.speed = MOVE_SPEED
    this.raycaster = new THREE.Raycaster()
    this.tempUp = new THREE.Vector3(0, 1, 0)
    this.tempQuaternion = new THREE.Quaternion()
    this.tempTangent = new THREE.Vector3()
    this.sharedPulseGeometry = new THREE.CylinderGeometry(1, 1, 1, 6, 1)
    this.forwardSignalColor = new THREE.Color(COLORS.forward)
    this.backwardSignalColor = new THREE.Color(COLORS.backward)
    this.trainingCueText = 'Forward batch + gradientes Excel + actualizacion de pesos'

    this.animate = this.animate.bind(this)
    this.handleResize = this.handleResize.bind(this)
    this.handleKeyDown = this.handleKeyDown.bind(this)
    this.handleKeyUp = this.handleKeyUp.bind(this)
    this.handleCanvasClick = this.handleCanvasClick.bind(this)
    this.handlePointerLockChange = this.handlePointerLockChange.bind(this)
    this.handleMouseMove = this.handleMouseMove.bind(this)
    this.handleWheel = this.handleWheel.bind(this)
  }

  mount({ viewportEl }) {
    if (!viewportEl) {
      return
    }

    this.viewportEl = viewportEl
    this.createScene()
    this.attachListeners()
    this.handleResize()
    this.applyCurrentModelToScene()
    this.emitState()
    this.animate()
  }

  unmount() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }

    if (document.pointerLockElement === this.renderer?.domElement) {
      document.exitPointerLock?.()
    }

    this.removeListeners()
    this.disposeScene()
    document.body.style.cursor = ''
  }

  createScene() {
    this.viewportEl.innerHTML = ''
    this.tourAnchors.clear()
    this.tourShots = []

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(COLORS.background)
    this.scene.fog = new THREE.FogExp2(COLORS.background, 0.0055)

    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      300,
    )
    this.camera.position.set(this.camX, this.camY, this.camZ)
    this.camera.lookAt(0, 0, 0)

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    this.renderer.setSize(window.innerWidth, window.innerHeight)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.24
    this.renderer.domElement.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;z-index:1;'
    this.viewportEl.appendChild(this.renderer.domElement)
    this.createPostProcessing()

    this.environmentGroup = new THREE.Group()
    this.scene.add(this.environmentGroup)
    this.createRoom()
    this.createDecisionSurface()
    this.networkGroup = new THREE.Group()
    this.scene.add(this.networkGroup)
    this.buildNetworkObjects()
    this.registerTourAnchor('entryWide', CAMERA_ENTRY.x, CAMERA_ENTRY.y + 0.4, CAMERA_ENTRY.z + 1.5)
    this.registerTourAnchor('entrySettle', 0, 3.2, 13.4)
    this.registerTourAnchor('entryFocus', 0, 1.5, 4.4)
    this.registerTourAnchor('networkWide', 8.8, 4.3, 12.8)
    this.registerTourAnchor('networkFree', CAMERA_DEFAULT.x, CAMERA_DEFAULT.y, CAMERA_DEFAULT.z)
    this.registerTourAnchor('networkFocus', 0, 1.3, 0)
    this.buildTourShots()
    this.updateDecisionSurface()
  }

  createPostProcessing() {
    this.composer = new EffectComposer(this.renderer)
    this.renderPass = new RenderPass(this.scene, this.camera)
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.28,
      0.2,
      0.72,
    )

    this.composer.addPass(this.renderPass)
    this.composer.addPass(this.bloomPass)
  }

  createRoom() {
    const { width, height, depth } = ROOM_DIMENSIONS
    const materials = {
      floor: new THREE.MeshStandardMaterial({
        color: COLORS.roomFloor,
        roughness: 0.92,
        metalness: 0.12,
        side: THREE.DoubleSide,
      }),
      ceiling: new THREE.MeshStandardMaterial({
        color: COLORS.roomCeiling,
        roughness: 0.8,
        metalness: 0.18,
        side: THREE.DoubleSide,
      }),
      back: new THREE.MeshStandardMaterial({
        color: COLORS.roomBack,
        roughness: 0.88,
        metalness: 0.16,
        side: THREE.DoubleSide,
      }),
      side: new THREE.MeshStandardMaterial({
        color: COLORS.roomSide,
        roughness: 0.85,
        metalness: 0.18,
        side: THREE.DoubleSide,
      }),
      panel: new THREE.MeshStandardMaterial({
        color: COLORS.roomPanel,
        roughness: 0.64,
        metalness: 0.3,
      }),
      trim: new THREE.MeshStandardMaterial({
        color: COLORS.roomTrim,
        roughness: 0.34,
        metalness: 0.68,
      }),
      console: new THREE.MeshStandardMaterial({
        color: 0x1a2732,
        roughness: 0.5,
        metalness: 0.38,
      }),
      detail: new THREE.MeshStandardMaterial({
        color: 0x233846,
        roughness: 0.56,
        metalness: 0.28,
      }),
      glass: new THREE.MeshPhysicalMaterial({
        color: COLORS.roomGlass,
        transparent: true,
        opacity: 0.08,
        roughness: 0.18,
        metalness: 0.08,
        side: THREE.DoubleSide,
      }),
      warning: new THREE.MeshStandardMaterial({
        color: 0x433521,
        emissive: COLORS.roomWarning,
        emissiveIntensity: 0.12,
        roughness: 0.4,
        metalness: 0.45,
      }),
    }

    this.createWall(width, depth, materials.floor, Math.PI / 2, 0, 0, 0, -height / 2, 0)
    this.createWall(width, depth, materials.ceiling, -Math.PI / 2, 0, 0, 0, height / 2, 0)
    this.createWall(width, height, materials.back, 0, 0, 0, 0, 0, -depth / 2)
    this.createWall(width, height, materials.back, 0, Math.PI, 0, 0, 0, depth / 2)
    this.createWall(depth, height, materials.side, 0, Math.PI / 2, 0, -width / 2, 0, 0)
    this.createWall(depth, height, materials.side, 0, -Math.PI / 2, 0, width / 2, 0, 0)

    this.createRoomFrame(materials, width, height, depth)
    this.createFloorDeck(materials, width, height, depth)
    this.createCornerColumns(materials, width, height, depth)
    this.createCeilingRig(materials, width, height, depth)
    this.createWallConsoles(materials, width, depth)
    this.createWallLightPanels(materials, width, height, depth)
    this.createWallShelves(materials, width, depth)
    this.createWallDecorClusters(materials, width, depth)
    this.createHeroWall(materials, width, height, depth)
    this.createResearchPanels(width, depth)
    this.createServerAisle(materials, width)
    this.createWorkbenchZone(materials, width, depth)
    this.createPrototypeBay(materials, width, depth)
    this.createFloorBots(materials, width, height, depth)

    this.scene.add(new THREE.AmbientLight(COLORS.ambient, 1.58))

    const directionalLight = new THREE.DirectionalLight(COLORS.directional, 1.58)
    directionalLight.position.set(6, 12, 8)
    this.scene.add(directionalLight)

    const fillLight = new THREE.DirectionalLight(0xa8d8ff, 0.86)
    fillLight.position.set(-8, 5, -10)
    this.scene.add(fillLight)

    const warmLight = new THREE.DirectionalLight(0xffc27c, 0.42)
    warmLight.position.set(8, 4, -6)
    this.scene.add(warmLight)

    this.heroSpotlight = new THREE.SpotLight(COLORS.hologramHigh, 1.35, 60, 0.42, 0.44, 1.2)
    this.heroSpotlight.position.set(0, height / 2 - 1.8, 8)
    this.heroSpotlight.target.position.set(0, 0.4, -1.5)
    this.scene.add(this.heroSpotlight)
    this.scene.add(this.heroSpotlight.target)

    this.addLabLight(new THREE.Vector3(0, height / 2 - 1.8, 0), COLORS.roomAccent, 0.96, 0.24, 22)
    this.addLabLight(new THREE.Vector3(0, -2.2, 0), COLORS.hologramHigh, 0.52, 0.16, 16)
    this.addLabLight(new THREE.Vector3(0, 1.8, -depth / 2 + 6), COLORS.hologramEdge, 0.24, 0.08, 12)
  }

  createRoomFrame(materials, width, height, depth) {
    const edge = 0.16
    const inset = 0.24
    const halfWidth = width / 2 - inset
    const halfHeight = height / 2 - inset
    const halfDepth = depth / 2 - inset

    ;[-halfHeight, halfHeight].forEach((y) => {
      ;[-halfDepth, halfDepth].forEach((z) => {
        this.createBox(width - 0.4, edge, edge, materials.trim, 0, y, z)
      })
      ;[-halfWidth, halfWidth].forEach((x) => {
        this.createBox(edge, edge, depth - 0.4, materials.trim, x, y, 0)
      })
    })

    ;[-halfWidth, halfWidth].forEach((x) => {
      ;[-halfDepth, halfDepth].forEach((z) => {
        this.createBox(edge, height - 0.4, edge, materials.trim, x, 0, z)
      })
    })

    ;[-1, 1].forEach((direction) => {
      this.createBox(width - 8, 0.12, 0.35, materials.detail, 0, 0, direction * (depth / 2 - 0.5))
      this.createBox(0.35, 0.12, depth - 8, materials.detail, direction * (width / 2 - 0.5), 0, 0)
    })
  }

  createFloorDeck(materials, width, height, depth) {
    const floorY = -height / 2 + 0.01

    ;[8.5, 13.8].forEach((size, index) => {
      const y = floorY + 0.04 + index * 0.015
      this.createBox(size, 0.03, 0.08, materials.detail, 0, y, size / 2)
      this.createBox(size, 0.03, 0.08, materials.detail, 0, y, -size / 2)
      this.createBox(0.08, 0.03, size, materials.detail, size / 2, y, 0)
      this.createBox(0.08, 0.03, size, materials.detail, -size / 2, y, 0)
    })

    ;[-1, 1].forEach((direction) => {
      this.createEmissiveStrip(
        width - 8,
        0.05,
        0.12,
        0,
        floorY + 0.06,
        direction * (depth / 2 - 2.1),
        0,
        0,
        0,
        COLORS.roomAccent,
        0.12,
        0.04,
      )
      this.createEmissiveStrip(
        0.12,
        0.05,
        depth - 8,
        direction * (width / 2 - 2.1),
        floorY + 0.06,
        0,
        0,
        0,
        0,
        COLORS.roomAccent,
        0.12,
        0.04,
      )
    })
  }

  createCornerColumns(materials, width, height, depth) {
    const offsetX = width / 2 - 1.2
    const offsetZ = depth / 2 - 1.2

    ;[-1, 1].forEach((xDirection) => {
      ;[-1, 1].forEach((zDirection) => {
        const x = xDirection * offsetX
        const z = zDirection * offsetZ

        this.createBox(1.2, height - 0.8, 1.2, materials.console, x, 0, z)
        this.createBox(0.5, 2.2, 0.3, materials.warning, x - xDirection * 0.42, 7, z)
        this.createEmissiveStrip(
          0.12,
          height * 0.72,
          0.12,
          x - xDirection * 0.48,
          -1.4,
          z - zDirection * 0.48,
          0,
          0,
          0,
          COLORS.roomAccent,
          0.16,
          0.05,
        )
      })
    })
  }

  createCeilingRig(materials, width, height, depth) {
    const ceilingY = height / 2 - 0.75

    this.createBox(width - 6, 0.45, 1.8, materials.panel, 0, ceilingY, 9.5)
    this.createBox(1.6, 0.35, depth - 7, materials.panel, -9.5, ceilingY - 0.3, 0)
    this.createBox(1.6, 0.35, depth - 7, materials.panel, 9.5, ceilingY - 0.3, 0)

    ;[-8.2, 8.2].forEach((x) => {
      ;[-8.2, 8.2].forEach((z) => {
        this.createEmissiveStrip(
          7.6,
          0.12,
          0.24,
          x,
          height / 2 - 1.1,
          z,
          0,
          0,
          0,
          COLORS.roomAccent,
          0.18,
          0.06,
        )
        this.addLabLight(
          new THREE.Vector3(x, height / 2 - 1.45, z),
          COLORS.point,
          0.72,
          0.2,
          12,
        )
      })
    })

    this.createBox(14.8, 0.12, 0.22, materials.trim, 0, height / 2 - 1.7, 0)
    this.createBox(0.22, 0.12, 14.8, materials.trim, 0, height / 2 - 1.7, 0)
  }

  createWallConsoles(materials, width) {
    const leftX = -width / 2 + 0.9
    const rightX = width / 2 - 0.9

    ;[-10, 0, 10].forEach((z, index) => {
      this.createConsoleStack(leftX, -11.1, z, Math.PI / 2, materials, `LAB-${index + 1}`, 'Training node')
      this.createConsoleStack(rightX, -11.1, z, -Math.PI / 2, materials, `BAY-${index + 1}`, 'Inference rack')
    })
  }

  createWallLightPanels(materials, width, height) {
    const topY = height / 2 - 4.4
    const sideX = width / 2 - 0.28

    ;[-10.5, 0, 10.5].forEach((z) => {
      this.createEmissiveStrip(0.18, 5.4, 1.7, -sideX, topY - 2.6, z, 0, 0, 0, COLORS.roomAccent, 0.24, 0.08)
      this.createEmissiveStrip(0.18, 5.4, 1.7, sideX, topY - 2.6, z, 0, 0, 0, COLORS.roomWarning, 0.16, 0.05)
    })
  }

  createWallShelves(materials, width) {
    const leftX = -width / 2 + 1.15
    const rightX = width / 2 - 1.15

    this.createShelfColumn(leftX, -10.7, -13.5, Math.PI / 2, materials, 'hardware')
    this.createShelfColumn(leftX, -10.7, 13.5, Math.PI / 2, materials, 'compute')
    this.createShelfColumn(rightX, -10.7, -13.5, -Math.PI / 2, materials, 'robotics')
    this.createShelfColumn(rightX, -10.7, 13.5, -Math.PI / 2, materials, 'coffee')
  }

  createShelfColumn(px, py, pz, ry, materials, theme) {
    this.createBox(0.32, 8.6, 3.8, materials.trim, px, py, pz, 0, ry, 0)
    this.createBox(0.32, 8.6, 3.8, materials.trim, px, py, pz + 0.98, 0, ry, 0)

    ;[-3, -0.8, 1.4, 3.6].forEach((offset) => {
      this.createBox(1.9, 0.16, 3.4, materials.panel, px, py + offset, pz + 0.49, 0, ry, 0)
    })

    if (theme === 'hardware') {
      this.createRamStick(px, py - 3, pz + 0.15, materials)
      this.createChipDisplay(px, py - 0.85, pz + 0.55, materials)
      this.createMiniTower(px, py + 1.45, pz + 0.55, materials)
      this.createStorageCrate(px, py + 3.6, pz + 0.55, materials)
    }

    if (theme === 'compute') {
      this.createServerMini(px, py - 3, pz + 0.55, materials)
      this.createServerMini(px, py - 0.8, pz + 0.55, materials)
      this.createServerMini(px, py + 1.4, pz + 0.55, materials)
      this.createStorageCrate(px, py + 3.6, pz + 0.55, materials)
    }

    if (theme === 'robotics') {
      this.createPrototypeRobot(px, py - 2.9, pz + 0.5, materials, 0.4)
      this.createPrototypeRobot(px, py - 0.75, pz + 0.5, materials, 1.4)
      this.createStorageCrate(px, py + 1.5, pz + 0.55, materials)
      this.createDeskLamp(px, py + 3.65, pz + 0.55, materials)
    }

    if (theme === 'coffee') {
      this.createCoffeeCup(px - 0.15, py - 3.05, pz + 0.5, materials)
      this.createStorageCrate(px, py - 0.8, pz + 0.55, materials)
      this.createMiniTower(px, py + 1.4, pz + 0.55, materials)
      this.createDeskLamp(px, py + 3.65, pz + 0.55, materials)
    }
  }

  createBackWallDisplays(materials, width, depth) {
    const backZ = -depth / 2 + 1.2

    this.createDisplayBench(-8.4, -10.8, backZ, materials, 'GPU ARRAY', '#8ef0ff')
    this.createDisplayBench(0, -10.8, backZ, materials, 'TRAINING TOOLS', '#ffd38a')
    this.createDisplayBench(8.4, -10.8, backZ, materials, 'ROBOT BAY', '#8ef0ff')
  }

  createDisplayBench(px, py, pz, materials, title, accent) {
    this.createBox(6.2, 2.8, 1.4, materials.panel, px, py + 1.4, pz)
    this.createBox(6.7, 0.22, 1.8, materials.trim, px, py + 2.9, pz)
    this.createBox(6.7, 0.22, 1.8, materials.trim, px, py - 0.1, pz)

    const texture = buildResearchPanelTexture(title, 'Room assets', ['Visible props and hardware', 'Ambient activity and glow', 'Live chamber identity'], accent)
    texture.colorSpace = THREE.SRGBColorSpace

    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(4.9, 1.6),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.97 }),
    )
    screen.position.set(px, py + 1.55, pz + 0.74)
    this.environmentGroup.add(screen)

    this.createStorageCrate(px - 1.55, py + 0.15, pz + 0.2, materials)
    this.createMiniTower(px, py + 0.25, pz + 0.2, materials)
    this.createDeskLamp(px + 1.45, py + 0.18, pz + 0.18, materials)
  }

  createWallDecorClusters(materials, width, depth) {
    const leftX = -width / 2 + 2.4
    const rightX = width / 2 - 2.4
    const creditsZ = -depth / 2 + 0.82

    this.createDecorCluster(leftX, -12, -6.4, Math.PI / 2, materials, '#8ef0ff')
    this.createDecorCluster(leftX, -12, 6.6, Math.PI / 2, materials, '#ffd38a')
    this.createDecorCluster(rightX, -12, -6.4, -Math.PI / 2, materials, '#7af0b2')
    this.createDecorCluster(rightX, -12, 6.6, -Math.PI / 2, materials, '#ff8ea1')

    this.createCreditsPanel(0, 7.9, creditsZ, 0)
  }

  createDecorCluster(px, py, pz, ry, materials, accent) {
    const accentMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color(accent),
      emissive: new THREE.Color(accent),
      emissiveIntensity: 0.1,
      roughness: 0.34,
      metalness: 0.54,
    })

    this.createBox(0.78, 0.78, 0.78, accentMaterial, px, py + 0.55, pz, 0.25, ry + 0.4, 0.1)
    this.createBox(0.34, 1.45, 0.34, materials.trim, px, py - 0.6, pz)
    this.createBox(1.4, 0.08, 1.4, materials.panel, px, py - 1.35, pz)
    this.createRamStick(px + 0.7, py - 0.5, pz + 0.2, materials)
    this.createStorageCrate(px - 0.76, py - 0.8, pz + 0.12, materials)
    this.addLabLight(new THREE.Vector3(px, py + 0.2, pz), accent, 0.18, 0.05, 4.5)
  }

  createColorBanner(px, py, pz, ry, accent, shadow) {
    const texture = buildResearchPanelTexture('LAB DETAIL', 'Visual layer', ['Hardware exhibits', 'Ambient color coding', 'Room personality'], accent)
    texture.colorSpace = THREE.SRGBColorSpace

    const backing = new THREE.Mesh(
      new THREE.BoxGeometry(5.1, 2.2, 0.18),
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(shadow),
        roughness: 0.45,
        metalness: 0.3,
      }),
    )
    backing.position.set(px, py, pz)
    backing.rotation.y = ry
    this.environmentGroup.add(backing)

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(4.8, 1.9),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.98 }),
    )
    plane.position.set(px, py, pz - 0.1)
    plane.rotation.y = ry
    this.environmentGroup.add(plane)
  }

  createCreditsPanel(px, py, pz, ry) {
    this.registerTourAnchor('creditsWide', 0, 4.2, -16.8)
    this.registerTourAnchor('creditsClose', 0, 7.2, -14.4)
    this.registerTourAnchor('creditsFocus', 0, 7.9, pz)

    const texture = buildCreditsTexture(['Jhonier Becerra', 'Jorge Chicaiza'])
    texture.colorSpace = THREE.SRGBColorSpace

    const backing = new THREE.Mesh(
      new THREE.BoxGeometry(10.2, 3.05, 0.16),
      new THREE.MeshStandardMaterial({
        color: 0x162530,
        roughness: 0.48,
        metalness: 0.3,
      }),
    )
    backing.position.set(px, py, pz)
    backing.rotation.y = ry
    this.environmentGroup.add(backing)

    const bezel = new THREE.Mesh(
      new THREE.BoxGeometry(9.9, 2.78, 0.05),
      new THREE.MeshStandardMaterial({
        color: 0x223746,
        roughness: 0.28,
        metalness: 0.68,
        emissive: COLORS.roomAccent,
        emissiveIntensity: 0.025,
      }),
    )
    bezel.position.set(px, py, pz - 0.04)
    bezel.rotation.y = ry
    this.environmentGroup.add(bezel)

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(9.55, 2.5),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.95,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: false,
      }),
    )
    plane.position.set(px, py, pz + 0.13)
    plane.rotation.y = ry
    plane.renderOrder = 1200
    this.environmentGroup.add(plane)
  }

  createHeroWall(materials, width, height, depth) {
    const z = depth / 2 - 0.72
    const y = 0.8

    this.createBox(21.5, 12.2, 0.58, materials.panel, 0, y, z)
    this.createBox(23.2, 0.28, 0.72, materials.trim, 0, y + 6.18, z)
    this.createBox(23.2, 0.28, 0.72, materials.trim, 0, y - 6.18, z)
    this.createBox(0.32, 12.6, 0.72, materials.trim, -11.58, y, z)
    this.createBox(0.32, 12.6, 0.72, materials.trim, 11.58, y, z)

    this.createEmissiveStrip(15.8, 0.12, 0.14, 0, y + 5.4, z + 0.36, 0, 0, 0, COLORS.roomAccent, 0.26, 0.08)
    this.createEmissiveStrip(9.6, 0.08, 0.14, 0, y - 4.95, z + 0.36, 0, 0, 0, COLORS.hologramEdge, 0.18, 0.05)

    this.createDynamicDisplayPanel({
      px: 0,
      py: y + 1.2,
      pz: z + 0.42,
      ry: Math.PI,
      width: 11.6,
      height: 4.8,
      accent: '#8ef0ff',
      title: 'Neural Command Wall',
      mode: 'hero',
    })

    this.createDynamicDisplayPanel({
      px: -7.8,
      py: y - 1.8,
      pz: z + 0.42,
      ry: Math.PI,
      width: 3.4,
      height: 1.9,
      accent: '#ffd38a',
      title: 'Loss Trace',
      mode: 'aux',
    })

    this.createDynamicDisplayPanel({
      px: 7.8,
      py: y - 1.8,
      pz: z + 0.42,
      ry: Math.PI,
      width: 3.4,
      height: 1.9,
      accent: '#7af0b2',
      title: 'Inference Grid',
      mode: 'aux',
    })

    ;[-9, -3.5, 3.5, 9].forEach((x, index) => {
      this.createHeroCanister(x, y - 4.95, z + 0.55, materials, index % 2 === 0 ? COLORS.roomAccent : COLORS.hologramEdge)
    })
  }

  createDynamicDisplayPanel({ px, py, pz, ry, width, height, accent, title, mode }) {
    const display = buildDisplayPanelTexture(title, accent)
    const backing = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.34, height + 0.34, 0.2),
      new THREE.MeshStandardMaterial({
        color: 0x162530,
        roughness: 0.38,
        metalness: 0.36,
      }),
    )
    backing.position.set(px, py, pz)
    backing.rotation.y = ry
    this.environmentGroup.add(backing)

    const bezel = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.16, height + 0.16, 0.08),
      new THREE.MeshStandardMaterial({
        color: 0x223746,
        roughness: 0.24,
        metalness: 0.72,
        emissive: new THREE.Color(accent),
        emissiveIntensity: 0.06,
      }),
    )
    bezel.position.set(px, py, pz + 0.08)
    bezel.rotation.y = ry
    this.environmentGroup.add(bezel)

    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      new THREE.MeshBasicMaterial({
        map: display.texture,
        transparent: true,
        opacity: 0.98,
      }),
    )
    panel.position.set(px, py, pz + 0.13)
    panel.rotation.y = ry
    this.environmentGroup.add(panel)

    this.displayPanels.push({
      ...display,
      panel,
      mode,
      accent,
    })

    renderDisplayPanelTexture(this.displayPanels[this.displayPanels.length - 1], {
      tick: 0,
      loss: 0.5,
      epoch: 0,
      outputs: createPlaceholderOutputs(),
      activity: mode === 'hero' ? 0.2 : 0.12,
    })
  }

  createHeroCanister(px, py, pz, materials, accent) {
    const shell = new THREE.Mesh(
      new THREE.CylinderGeometry(0.62, 0.62, 1.8, 20, 1, true),
      new THREE.MeshPhysicalMaterial({
        color: COLORS.roomGlass,
        transparent: true,
        opacity: 0.14,
        roughness: 0.08,
        metalness: 0.02,
        transmission: 0.42,
      }),
    )
    shell.position.set(px, py, pz)
    this.environmentGroup.add(shell)

    this.createBox(0.78, 0.12, 0.78, materials.trim, px, py + 0.92, pz)
    this.createBox(0.78, 0.12, 0.78, materials.trim, px, py - 0.92, pz)
    this.createEmissiveStrip(0.12, 1.38, 0.12, px, py, pz, 0, 0, 0, accent, 0.16, 0.05)
  }

  createResearchPanels(width, depth) {
    this.createReferencePanel(
      -width / 2 + 0.45,
      5.4,
      -10.5,
      Math.PI / 2,
      'McCulloch-Pitts',
      '1943 - formal neuron model',
      ['Threshold logic units', 'First abstract neural model', 'Foundation for connectionism'],
      '#7bd5ff',
    )
    this.createReferencePanel(
      -width / 2 + 0.45,
      5.4,
      9.5,
      Math.PI / 2,
      'Perceptron',
      '1958 - Rosenblatt',
      ['Learnable linear classifier', 'Weights adapt from data', 'Limits exposed by XOR'],
      '#a9e7ff',
    )
    this.createReferencePanel(
      width / 2 - 0.45,
      5.4,
      -10.5,
      -Math.PI / 2,
      'Backpropagation',
      '1986 - Rumelhart, Hinton, Williams',
      ['Gradient-based learning', 'Multilayer training', 'Errors flow backward'],
      '#ffbe76',
    )
    this.createReferencePanel(
      width / 2 - 0.45,
      5.4,
      9.5,
      -Math.PI / 2,
      'XOR Chamber',
      'Current experiment',
      ['Non linear separability', 'Hidden layers required', 'Decision field below'],
      '#ffd89c',
    )

    this.createReferencePanel(
      0,
      8.2,
      depth / 2 - 0.45,
      Math.PI,
      'Neural Research Cube',
      'Interactive observation bay',
      ['Forward signal and backprop', 'Live XOR decision surface', 'Walk inside the model'],
      '#8ef0ff',
      7.4,
      3.4,
    )
  }

  createTrainingCore(materials) {
    const ringMaterial = new THREE.MeshStandardMaterial({
      color: 0x173646,
      emissive: COLORS.hologramMid,
      emissiveIntensity: 0.16,
      roughness: 0.28,
      metalness: 0.72,
    })
    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: COLORS.roomGlass,
      transparent: true,
      opacity: 0.09,
      roughness: 0.08,
      metalness: 0.02,
      transmission: 0.35,
      side: THREE.DoubleSide,
    })

    this.createBox(11.5, 0.32, 11.5, materials.trim, 0, -5.98, 0)
    this.createBox(8.9, 0.18, 8.9, materials.panel, 0, -5.72, 0)

    const lowerRing = new THREE.Mesh(
      new THREE.TorusGeometry(5.6, 0.18, 18, 80),
      ringMaterial.clone(),
    )
    lowerRing.rotation.x = Math.PI / 2
    lowerRing.position.set(0, -1.2, 0)
    this.environmentGroup.add(lowerRing)
    this.labCoreRings.push({ mesh: lowerRing, speed: 0.3, axis: 'y', bob: 0.08, phase: 0 })

    const upperRing = new THREE.Mesh(
      new THREE.TorusGeometry(3.8, 0.12, 18, 72),
      ringMaterial.clone(),
    )
    upperRing.rotation.set(Math.PI / 2.9, 0, 0.7)
    upperRing.position.set(0, 1.5, 0)
    this.environmentGroup.add(upperRing)
    this.labCoreRings.push({ mesh: upperRing, speed: -0.42, axis: 'z', bob: 0.12, phase: 1.5 })

    const shell = new THREE.Mesh(new THREE.CylinderGeometry(4.9, 4.9, 8.6, 10, 1, true), glassMaterial)
    shell.position.set(0, -1.4, 0)
    this.environmentGroup.add(shell)

    const capTop = new THREE.Mesh(new THREE.CircleGeometry(4.9, 48), glassMaterial)
    capTop.rotation.x = -Math.PI / 2
    capTop.position.set(0, 2.9, 0)
    this.environmentGroup.add(capTop)

    const capBottom = new THREE.Mesh(new THREE.CircleGeometry(4.9, 48), glassMaterial)
    capBottom.rotation.x = Math.PI / 2
    capBottom.position.set(0, -5.7, 0)
    this.environmentGroup.add(capBottom)

    ;[-1, 1].forEach((direction) => {
      this.createBox(0.18, 8.2, 0.18, materials.trim, direction * 3.9, -1.4, direction * 3.9)
      this.createBox(0.18, 8.2, 0.18, materials.trim, direction * 3.9, -1.4, -direction * 3.9)
    })

    ;[
      [-8.8, -5.8],
      [8.8, -5.8],
      [-8.8, 5.8],
      [8.8, 5.8],
    ].forEach(([x, z], index) => {
      this.createCableRun(x, -6.3, z, x * 0.28, -4.8, z * 0.28, 0, -4.1, 0, index)
    })
  }

  createServerAisle(materials, width) {
    const leftX = -width / 2 + 3.6

    this.registerTourAnchor('rackWide', -11.8, 2.9, -14.2)
    this.registerTourAnchor('rackClose', -17.1, 1.8, -4.6)
    this.registerTourAnchor('rackFocus', leftX - 0.6, -7.9, -2.2)
    this.registerTourAnchor('rackDetail', leftX + 0.8, -8.9, -4)

    ;[-11, -4, 3, 10].forEach((z, rackIndex) => {
      this.createServerRack(leftX, -11.1, z, Math.PI / 2, materials, `TRAIN-${rackIndex + 1}`)
    })
  }

  createServerRack(px, py, pz, ry, materials, label) {
    this.createBox(2.2, 7.8, 4.3, materials.console, px, py, pz, 0, ry, 0)
    this.createBox(2.4, 0.2, 4.5, materials.trim, px, py + 4, pz, 0, ry, 0)
    this.createBox(2.4, 0.2, 4.5, materials.trim, px, py - 4, pz, 0, ry, 0)

    for (let row = 0; row < 5; row += 1) {
      const y = py + 2.7 - row * 1.35
      this.createBox(0.24, 0.82, 3.3, materials.detail, px + 0.88, y, pz, 0, ry, 0)
      this.createEmissiveStrip(
        0.08,
        0.18,
        2.6,
        px + 1.06,
        y,
        pz,
        0,
        ry,
        0,
        row % 2 === 0 ? COLORS.forward : COLORS.backward,
        0.14,
        0.04,
      )
    }

    const fanOffsets = [-1.18, 0, 1.18]

    fanOffsets.forEach((offset, index) => {
      const fan = new THREE.Mesh(
        new THREE.CylinderGeometry(0.42, 0.42, 0.14, 24),
        new THREE.MeshStandardMaterial({
          color: 0x0d141b,
          emissive: 0x163346,
          emissiveIntensity: 0.18,
          roughness: 0.42,
          metalness: 0.68,
        }),
      )
      fan.rotation.z = Math.PI / 2
      fan.position.set(px - 1.02, py + 2.65 - index * 2.35, pz + offset)
      this.environmentGroup.add(fan)
      this.labFans.push({ mesh: fan, speed: 0.22 + index * 0.05 })
    })

    const texture = buildResearchPanelTexture(label, 'GPU cluster', ['1024 tensor ops', 'Thermal stable', 'Data stream active'], '#8ef0ff')
    texture.colorSpace = THREE.SRGBColorSpace

    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(1.8, 1),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.95 }),
    )
    screen.position.set(px + 1.12, py + 2.8, pz)
    screen.rotation.y = ry
    this.environmentGroup.add(screen)
  }

  createWorkbenchZone(materials, width, depth) {
    const x = width / 2 - 7.2
    const z = depth / 2 - 9.2

    this.registerTourAnchor('workbenchWide', 9.8, 2.6, 14.9)
    this.registerTourAnchor('workbenchClose', 14.5, -9.1, 12.2)
    this.registerTourAnchor('workbenchFocus', x, -11.4, z)
    this.registerTourAnchor('workbenchDetail', x - 0.9, -11.2, z - 0.2)

    this.createBox(6.4, 0.28, 3.2, materials.panel, x, -12.5, z)
    this.createBox(0.24, 3.2, 0.24, materials.trim, x - 2.8, -13.4, z - 1.2)
    this.createBox(0.24, 3.2, 0.24, materials.trim, x + 2.8, -13.4, z - 1.2)
    this.createBox(0.24, 3.2, 0.24, materials.trim, x - 2.8, -13.4, z + 1.2)
    this.createBox(0.24, 3.2, 0.24, materials.trim, x + 2.8, -13.4, z + 1.2)

    this.createMonitor(x - 1.6, -11.05, z - 0.55, materials, 0.12)
    this.createMonitor(x + 0.2, -10.88, z - 0.1, materials, -0.05)
    this.createKeyboard(x - 0.9, -12.22, z + 0.32, materials)
    this.createCoffeeCup(x + 2.1, -12.05, z + 0.4, materials)
    this.createChipDisplay(x + 1.1, -11.9, z - 0.7, materials)

    const armBase = this.createBox(0.9, 0.5, 0.9, materials.console, x + 2.4, -12.25, z - 1.08)
    const arm = this.createBox(0.24, 2.6, 0.24, materials.trim, x + 2.4, -10.8, z - 1.08)
    const forearm = this.createBox(1.5, 0.18, 0.18, materials.trim, x + 1.86, -9.78, z - 1.08, 0, 0, 0.5)
    const probe = this.createBox(0.16, 0.8, 0.16, materials.warning, x + 1.16, -10.28, z - 1.08)

    this.labBots.push({
      base: armBase,
      body: arm,
      head: forearm,
      probe,
      anchor: new THREE.Vector3(x + 2.4, -12.25, z - 1.08),
      phase: 0.8,
      kind: 'arm',
    })
  }

  createMonitor(px, py, pz, materials, tilt) {
    const texture = buildResearchPanelTexture('Vision feed', 'Gradient watch', ['Loss pulse map', 'Neuron activation', 'Inference latency'], '#67d7ff')
    texture.colorSpace = THREE.SRGBColorSpace
    this.createBox(1.7, 1.06, 0.12, materials.trim, px, py, pz, tilt, 0.18, 0)
    this.createBox(0.12, 0.66, 0.12, materials.trim, px, py - 0.8, pz)
    this.createBox(0.82, 0.08, 0.46, materials.trim, px, py - 1.15, pz)

    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(1.46, 0.82),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.97 }),
    )
    screen.position.set(px, py, pz + 0.07)
    screen.rotation.set(tilt, 0.18, 0)
    this.environmentGroup.add(screen)
  }

  createKeyboard(px, py, pz, materials) {
    this.createBox(1.55, 0.08, 0.58, materials.detail, px, py, pz)

    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 6; column += 1) {
        this.createBox(
          0.15,
          0.03,
          0.1,
          materials.trim,
          px - 0.48 + column * 0.18,
          py + 0.05,
          pz - 0.14 + row * 0.14,
        )
      }
    }
  }

  createCoffeeCup(px, py, pz) {
    const cup = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.18, 0.42, 20),
      new THREE.MeshStandardMaterial({
        color: 0xc4d8e3,
        roughness: 0.62,
        metalness: 0.08,
      }),
    )
    cup.position.set(px, py, pz)
    this.environmentGroup.add(cup)

    const coffee = new THREE.Mesh(
      new THREE.CylinderGeometry(0.17, 0.17, 0.02, 20),
      new THREE.MeshBasicMaterial({ color: 0x3b2216 }),
    )
    coffee.position.set(px, py + 0.17, pz)
    this.environmentGroup.add(coffee)

    for (let index = 0; index < 4; index += 1) {
      const puff = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 10, 10),
        new THREE.MeshBasicMaterial({ color: 0xc8f5ff, transparent: true, opacity: 0.08 }),
      )
      puff.position.set(px + (index - 1.5) * 0.03, py + 0.36 + index * 0.16, pz)
      this.environmentGroup.add(puff)
      this.labSteam.push({ mesh: puff, baseY: puff.position.y, phase: index * 0.6 })
    }
  }

  createChipDisplay(px, py, pz, materials) {
    this.createBox(0.9, 0.12, 0.9, materials.warning, px, py, pz)
    this.createBox(0.58, 0.08, 0.58, materials.console, px, py + 0.1, pz)

    for (let index = 0; index < 4; index += 1) {
      this.createBox(0.1, 0.06, 0.22, materials.trim, px - 0.34 + index * 0.22, py + 0.08, pz + 0.42)
      this.createBox(0.1, 0.06, 0.22, materials.trim, px - 0.34 + index * 0.22, py + 0.08, pz - 0.42)
      this.createBox(0.22, 0.06, 0.1, materials.trim, px + 0.42, py + 0.08, pz - 0.34 + index * 0.22)
      this.createBox(0.22, 0.06, 0.1, materials.trim, px - 0.42, py + 0.08, pz - 0.34 + index * 0.22)
    }
  }

  createMiniTower(px, py, pz, materials) {
    this.createBox(0.66, 1.1, 0.54, materials.console, px, py, pz)
    this.createBox(0.14, 0.82, 0.42, materials.detail, px + 0.28, py, pz)

    ;[-0.22, 0, 0.22].forEach((offset, index) => {
      this.createEmissiveStrip(0.04, 0.08, 0.12, px + 0.34, py + 0.26 - index * 0.26, pz + offset, 0, 0, 0, index === 1 ? COLORS.backward : COLORS.forward, 0.12, 0.04)
    })
  }

  createStorageCrate(px, py, pz, materials) {
    this.createBox(0.78, 0.46, 0.58, materials.panel, px, py, pz)
    this.createBox(0.82, 0.08, 0.62, materials.trim, px, py + 0.24, pz)
    this.createBox(0.3, 0.06, 0.08, materials.warning, px, py + 0.03, pz + 0.26)
  }

  createDeskLamp(px, py, pz, materials) {
    this.createBox(0.44, 0.06, 0.44, materials.trim, px, py - 0.2, pz)
    this.createBox(0.08, 0.68, 0.08, materials.trim, px, py + 0.12, pz)
    this.createBox(0.08, 0.62, 0.08, materials.trim, px + 0.18, py + 0.45, pz, 0, 0, -0.55)
    this.createBox(0.34, 0.16, 0.26, materials.warning, px + 0.36, py + 0.72, pz)
    this.addLabLight(new THREE.Vector3(px + 0.56, py + 0.5, pz), COLORS.roomWarning, 0.22, 0.04, 5)
  }

  createServerMini(px, py, pz, materials) {
    this.createBox(0.76, 0.36, 0.72, materials.console, px, py, pz)
    this.createBox(0.1, 0.18, 0.54, materials.detail, px + 0.32, py, pz)
    this.createEmissiveStrip(0.04, 0.06, 0.22, px + 0.38, py + 0.08, pz - 0.14, 0, 0, 0, COLORS.forward, 0.1, 0.03)
    this.createEmissiveStrip(0.04, 0.06, 0.22, px + 0.38, py - 0.08, pz + 0.14, 0, 0, 0, COLORS.backward, 0.1, 0.03)
  }

  createPrototypeBay(materials, width, depth) {
    const x = width / 2 - 5.2
    const z = -depth / 2 + 6.2

    this.registerTourAnchor('robotBayWide', 12.6, 1.9, -16.4)
    this.registerTourAnchor('robotBayClose', 16.8, -9.2, -13.1)
    this.registerTourAnchor('robotBayFocus', x, -10.3, z)
    this.registerTourAnchor('robotBayDetail', x + 2.2, -7.1, z)

    this.createBox(7.2, 0.28, 1.6, materials.panel, x, -13.6, z)
    this.createBox(7.2, 0.28, 1.6, materials.panel, x, -10.8, z)
    this.createBox(7.2, 0.28, 1.6, materials.panel, x, -8, z)
    this.createBox(0.2, 7.4, 0.2, materials.trim, x - 3.3, -10.6, z)
    this.createBox(0.2, 7.4, 0.2, materials.trim, x + 3.3, -10.6, z)

    this.createPrototypeRobot(x - 2.3, -12.5, z, materials, 0)
    this.createPrototypeRobot(x, -9.7, z, materials, 1.2)
    this.createPrototypeRobot(x + 2.2, -6.95, z, materials, 2.1)
    this.createRamStick(x + 2.1, -12.6, z + 0.38, materials)
    this.createRamStick(x - 0.6, -9.8, z + 0.32, materials)
  }

  createPrototypeRobot(px, py, pz, materials, phase) {
    const body = this.createBox(0.72, 0.5, 0.52, materials.console, px, py, pz)
    const head = this.createBox(0.46, 0.36, 0.38, materials.trim, px, py + 0.48, pz)
    const eyeLeft = this.createBox(0.08, 0.08, 0.05, materials.warning, px - 0.1, py + 0.48, pz + 0.2)
    const eyeRight = this.createBox(0.08, 0.08, 0.05, materials.warning, px + 0.1, py + 0.48, pz + 0.2)

    ;[-1, 1].forEach((direction) => {
      this.createBox(0.08, 0.42, 0.08, materials.trim, px + direction * 0.18, py - 0.44, pz + 0.12)
      this.createBox(0.08, 0.42, 0.08, materials.trim, px + direction * 0.18, py - 0.44, pz - 0.12)
    })

    this.labBots.push({ body, head, eyeLeft, eyeRight, anchor: new THREE.Vector3(px, py, pz), phase, kind: 'robot' })
  }

  createRamStick(px, py, pz, materials) {
    this.createBox(0.22, 1.1, 0.08, materials.warning, px, py, pz, 0.22, 0.1, 0)

    for (let index = 0; index < 5; index += 1) {
      this.createBox(0.06, 0.12, 0.05, materials.trim, px, py + 0.36 - index * 0.18, pz + 0.06)
    }
  }

  createFloorBots(materials, width, height, depth) {
    const floorY = -height / 2 + 0.18
    const roombas = [
      {
        x: -width / 2 + 6.8,
        z: depth / 2 - 7.4,
        accent: COLORS.roomAccent,
        phase: 0.4,
      },
      {
        x: width / 2 - 7.6,
        z: -depth / 2 + 8.2,
        accent: COLORS.hologramEdge,
        phase: 2.1,
      },
    ]

    roombas.forEach((botConfig) => {
      const body = new THREE.Mesh(
        new THREE.CylinderGeometry(0.92, 1.02, 0.34, 32),
        new THREE.MeshStandardMaterial({
          color: 0x192733,
          roughness: 0.34,
          metalness: 0.68,
        }),
      )
      body.position.set(botConfig.x, floorY, botConfig.z)
      this.environmentGroup.add(body)

      const top = new THREE.Mesh(
        new THREE.CylinderGeometry(0.58, 0.62, 0.09, 24),
        new THREE.MeshStandardMaterial({
          color: 0x304b5b,
          emissive: botConfig.accent,
          emissiveIntensity: 0.12,
          roughness: 0.18,
          metalness: 0.82,
        }),
      )
      top.position.set(botConfig.x, floorY + 0.16, botConfig.z)
      this.environmentGroup.add(top)

      const eyeLeft = this.createBox(0.14, 0.08, 0.08, materials.warning, botConfig.x - 0.18, floorY + 0.04, botConfig.z + 0.74)
      const eyeRight = this.createBox(0.14, 0.08, 0.08, materials.warning, botConfig.x + 0.18, floorY + 0.04, botConfig.z + 0.74)
      const head = this.createBox(0.52, 0.04, 0.08, materials.trim, botConfig.x, floorY - 0.15, botConfig.z + 0.92)

      this.labBots.push({
        body,
        top,
        head,
        eyeLeft,
        eyeRight,
        anchor: new THREE.Vector3(botConfig.x, floorY, botConfig.z),
        phase: botConfig.phase,
        accent: botConfig.accent,
        kind: 'roomba',
      })
    })
  }

  createCableRun(x1, y1, z1, x2, y2, z2, x3, y3, z3, seed) {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x1, y1, z1),
      new THREE.Vector3(x2, y2, z2),
      new THREE.Vector3(x3, y3, z3),
    ])
    const cable = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 36, 0.11, 8, false),
      new THREE.MeshStandardMaterial({
        color: 0x18303d,
        emissive: seed % 2 === 0 ? COLORS.forward : COLORS.backward,
        emissiveIntensity: 0.08,
        roughness: 0.48,
        metalness: 0.68,
      }),
    )
    this.environmentGroup.add(cable)
  }

  createReferencePanel(px, py, pz, ry, title, subtitle, lines, accent, width = 5.4, height = 2.6) {
    const texture = buildResearchPanelTexture(title, subtitle, lines, accent)
    texture.colorSpace = THREE.SRGBColorSpace

    const backing = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.28, height + 0.28, 0.26),
      new THREE.MeshStandardMaterial({
        color: COLORS.roomPanel,
        roughness: 0.52,
        metalness: 0.28,
      }),
    )
    backing.position.set(px, py, pz)
    backing.rotation.y = ry
    this.environmentGroup.add(backing)

    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.98,
      }),
    )
    const normal = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), ry)
    panel.position.set(px, py, pz)
    panel.position.addScaledVector(normal, 0.14)
    panel.rotation.y = ry
    this.environmentGroup.add(panel)

    this.createEmissiveStrip(
      width * 0.82,
      0.06,
      0.06,
      panel.position.x,
      panel.position.y + height / 2 + 0.12,
      panel.position.z,
      0,
      ry,
      0,
      accent,
      0.14,
      0.05,
    )
  }

  createConsoleStack(px, py, pz, ry, materials, title, subtitle) {
    this.createBox(1.2, 4.8, 3.8, materials.console, px, py, pz, 0, ry, 0)
    this.createBox(1.5, 0.18, 4.15, materials.trim, px, py + 2.48, pz, 0, ry, 0)
    this.createBox(1.5, 0.18, 4.15, materials.trim, px, py - 2.48, pz, 0, ry, 0)
    this.createBox(0.18, 4.3, 0.42, materials.detail, px, py, pz - 1.2, 0, ry, 0)
    this.createBox(0.18, 4.3, 0.42, materials.detail, px, py, pz + 1.2, 0, ry, 0)

    const texture = buildResearchPanelTexture(title, subtitle, ['Live chamber status', 'Weights and activations', 'Observation interface'], '#8ef0ff')
    texture.colorSpace = THREE.SRGBColorSpace

    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(2.4, 1.4),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.96,
      }),
    )
    screen.position.set(px, py + 0.4, pz)
    screen.rotation.y = ry
    screen.position.addScaledVector(
      new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), ry),
      0.72,
    )
    this.environmentGroup.add(screen)

    this.createEmissiveStrip(
      0.08,
      3.5,
      0.08,
      px,
      py,
      pz - 1.5,
      0,
      ry,
      0,
      COLORS.roomAccent,
      0.15,
      0.05,
    )
  }

  createEmissiveStrip(width, height, depth, px, py, pz, rx, ry, rz, color, baseIntensity, pulseRange) {
    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: baseIntensity,
      roughness: 0.28,
      metalness: 0.74,
    })

    this.createBox(width, height, depth, material, px, py, pz, rx, ry, rz)
    this.labAccentMaterials.push({
      material,
      baseIntensity,
      pulseRange,
      speed: 1.05 + this.labAccentMaterials.length * 0.03,
      phase: this.labAccentMaterials.length * 0.7,
    })
  }

  addLabLight(position, color, baseIntensity, pulseRange, distance) {
    const light = new THREE.PointLight(color, baseIntensity, distance)
    light.position.copy(position)
    this.scene.add(light)
    this.labAccentLights.push({
      light,
      baseIntensity,
      pulseRange,
      speed: 1.1 + this.labAccentLights.length * 0.04,
      phase: this.labAccentLights.length * 0.9,
    })
  }

  createBox(width, height, depth, material, px, py, pz, rx = 0, ry = 0, rz = 0) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material)
    mesh.position.set(px, py, pz)
    mesh.rotation.set(rx, ry, rz)
    this.environmentGroup.add(mesh)
    return mesh
  }

  createWall(width, height, material, rx, ry, rz, px, py, pz) {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      material,
    )

    mesh.rotation.set(rx, ry, rz)
    mesh.position.set(px, py, pz)
    this.environmentGroup.add(mesh)
  }

  registerTourAnchor(id, x, y, z) {
    this.tourAnchors.set(id, new THREE.Vector3(x, y, z))
  }

  getTourAnchor(id) {
    return this.tourAnchors.get(id)?.clone() ?? new THREE.Vector3()
  }

  buildTourShots() {
    this.tourShots = [
      {
        id: 'entry',
        from: 'entryWide',
        to: 'entrySettle',
        lookFrom: 'entryFocus',
        lookTo: 'entryFocus',
        duration: 2.8,
        label: 'Ingreso al laboratorio',
        fovFrom: 60,
        fovTo: 58,
      },
      {
        id: 'robots',
        from: 'robotBayWide',
        to: 'robotBayClose',
        lookFrom: 'robotBayFocus',
        lookTo: 'robotBayDetail',
        duration: 3.2,
        label: 'Prototipos roboticos y bots de piso',
        fovFrom: 57,
        fovTo: 51,
      },
      {
        id: 'racks',
        from: 'rackWide',
        to: 'rackClose',
        lookFrom: 'rackFocus',
        lookTo: 'rackDetail',
        duration: 3,
        label: 'Racks y consolas de entrenamiento',
        fovFrom: 56,
        fovTo: 50,
      },
      {
        id: 'workbench',
        from: 'workbenchWide',
        to: 'workbenchClose',
        lookFrom: 'workbenchFocus',
        lookTo: 'workbenchDetail',
        duration: 3,
        label: 'PC, monitores y banco de trabajo',
        fovFrom: 56,
        fovTo: 49,
      },
      {
        id: 'credits',
        from: 'creditsWide',
        to: 'creditsClose',
        lookFrom: 'creditsFocus',
        lookTo: 'creditsFocus',
        duration: 2.7,
        label: 'Integrantes del proyecto',
        fovFrom: 54,
        fovTo: 47,
      },
      {
        id: 'network',
        from: 'networkWide',
        to: 'networkFree',
        lookFrom: 'networkFocus',
        lookTo: 'networkFocus',
        duration: 3.6,
        label: 'Red neuronal lista para explorar',
        fovFrom: 57,
        fovTo: 60,
      },
    ]
  }

  applyTourShot(shot, progress) {
    const eased = smoothstep01(progress)
    const from = this.getTourAnchor(shot.from)
    const to = this.getTourAnchor(shot.to)
    const lookFrom = this.getTourAnchor(shot.lookFrom)
    const lookTo = this.getTourAnchor(shot.lookTo)

    this.tourCameraPoint.lerpVectors(from, to, eased)
    this.tourLookPoint.lerpVectors(lookFrom, lookTo, eased)

    this.camX = this.tourCameraPoint.x
    this.camY = this.tourCameraPoint.y
    this.camZ = this.tourCameraPoint.z
    this.camera.position.copy(this.tourCameraPoint)
    this.camera.fov = lerp(shot.fovFrom ?? 60, shot.fovTo ?? 60, eased)
    this.camera.updateProjectionMatrix()
    this.camera.lookAt(this.tourLookPoint)
  }

  startGuidedTour() {
    if (!this.camera || this.tourShots.length === 0) {
      return
    }

    if (document.pointerLockElement === this.renderer?.domElement) {
      document.exitPointerLock?.()
    }

    this.entryVisible = false
    this.pointerLocked = false
    this.tourActive = true
    this.tourShotIndex = 0
    this.tourShotTime = 0
    this.tourLabel = this.tourShots[0].label
    this.lastFrameTime = 0
    this.clearSelection()
    this.trainingCueText = 'Recorrido guiado por el laboratorio'
    this.applyTourShot(this.tourShots[0], 0)
    this.emitState()
  }

  finishGuidedTour() {
    const finalShot = this.tourShots[this.tourShots.length - 1]

    if (finalShot) {
      this.applyTourShot(finalShot, 1)
      this.aimCameraAt(this.tourLookPoint.x, this.tourLookPoint.y, this.tourLookPoint.z)
    }

    this.tourActive = false
    this.tourShotIndex = 0
    this.tourShotTime = 0
    this.tourLabel = ''
    this.trainingCueText = 'Forward batch + gradientes Excel + actualizacion de pesos'
    this.emitState()
  }

  advanceGuidedTour(delta) {
    if (!this.tourActive) {
      return
    }

    const shot = this.tourShots[this.tourShotIndex]

    if (!shot) {
      this.finishGuidedTour()
      return
    }

    this.tourShotTime = Math.min(this.tourShotTime + delta, shot.duration)
    const progress = shot.duration > 0 ? this.tourShotTime / shot.duration : 1
    this.applyTourShot(shot, progress)

    if (progress < 1) {
      return
    }

    if (this.tourShotIndex >= this.tourShots.length - 1) {
      this.finishGuidedTour()
      return
    }

    this.tourShotIndex += 1
    this.tourShotTime = 0
    this.tourLabel = this.tourShots[this.tourShotIndex].label
    this.emitState()
  }

  getNeuronPosition(layerIndex, neuronIndex) {
    return new THREE.Vector3(
      layerIndex * LAYER_SPACING - ((LAYERS.length - 1) * LAYER_SPACING) / 2,
      (neuronIndex - (LAYERS[layerIndex] - 1) / 2) * NODE_SPACING,
      0,
    )
  }

  buildNetworkObjects() {
    this.neurons = []
    this.neuronMap = []
    this.nodePointMap = []
    this.connections = []
    this.weightSprites = []
    this.layerSprites = []

    const sphereGeometry = new THREE.SphereGeometry(0.84, 40, 40)

    for (let layerIndex = 0; layerIndex < LAYERS.length; layerIndex += 1) {
      this.neuronMap.push([])
      this.nodePointMap.push([])

      for (let neuronIndex = 0; neuronIndex < LAYERS[layerIndex]; neuronIndex += 1) {
        const position = this.getNeuronPosition(layerIndex, neuronIndex)
        const activation = this.engine.activations[layerIndex][neuronIndex]
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color().setHSL(0.58, 0.75, 0.22 + activation * 0.38),
          emissive: new THREE.Color().setHSL(0.58, 0.9, 0.07 + activation * 0.22),
          roughness: 0.28,
          metalness: 0.12,
        })
        const mesh = new THREE.Mesh(sphereGeometry, material)
        mesh.position.copy(position)
        this.networkGroup.add(mesh)

          const pointLight = new THREE.PointLight(COLORS.nodeLight, 0.78 + activation * 0.86, 5.8)
        pointLight.position.copy(position)
        this.networkGroup.add(pointLight)

        const neuronRecord = {
          mesh,
          material,
          light: pointLight,
          layer: layerIndex,
          index: neuronIndex,
          activation,
          position: position.clone(),
          selected: false,
          forwardFlash: 0,
          backwardFlash: 0,
        }

        this.nodePointMap[layerIndex].push(position.clone())
        this.neuronMap[layerIndex].push(neuronRecord)
        this.neurons.push(neuronRecord)
      }
    }

    for (let layerIndex = 0; layerIndex < LAYERS.length - 1; layerIndex += 1) {
      for (let fromIndex = 0; fromIndex < LAYERS[layerIndex]; fromIndex += 1) {
        for (let toIndex = 0; toIndex < LAYERS[layerIndex + 1]; toIndex += 1) {
          const weight = this.engine.weights[layerIndex][fromIndex][toIndex]
          const from = this.nodePointMap[layerIndex][fromIndex]
          const to = this.nodePointMap[layerIndex + 1][toIndex]
          const seed = layerIndex * 100 + fromIndex * 10 + toIndex
          const curve = this.makeCurve(from, to, seed)
          const base = this.makeBaseTube(curve, weight)

          this.networkGroup.add(base.mesh)
          this.connections.push({
            base,
            curve,
            weight,
            layer: layerIndex,
            fromIndex,
            toIndex,
            seed,
            pulses: [],
            forwardFlash: 0,
            backwardFlash: 0,
          })
        }
      }
    }

    this.connections.forEach((connection, connectionIndex) => {
      const absoluteWeight = Math.abs(connection.weight)
      const radius = 0.025 + absoluteWeight * 0.13

      for (let dashIndex = 0; dashIndex < DASH_COUNT; dashIndex += 1) {
        const material = new THREE.MeshBasicMaterial({
          color: getWeightColor3(connection.weight),
          transparent: true,
          opacity: 0,
        })
        const mesh = new THREE.Mesh(this.sharedPulseGeometry, material)

        this.networkGroup.add(mesh)
        connection.pulses.push({
          mesh,
          material,
          t: dashIndex * DASH_GAP + ((connectionIndex * 0.137) % 1),
          radius,
        })
      }
    })

    const weightLabelData = []

    for (let layerIndex = 0; layerIndex < LAYERS.length - 1; layerIndex += 1) {
      for (let fromIndex = 0; fromIndex < LAYERS[layerIndex]; fromIndex += 1) {
        for (let toIndex = 0; toIndex < LAYERS[layerIndex + 1]; toIndex += 1) {
          const weight = this.engine.weights[layerIndex][fromIndex][toIndex]
          const from = this.nodePointMap[layerIndex][fromIndex]
          const to = this.nodePointMap[layerIndex + 1][toIndex]
          const seed = layerIndex * 100 + fromIndex * 10 + toIndex
          const totalInLayer = LAYERS[layerIndex] * LAYERS[layerIndex + 1]
          const indexInLayer = fromIndex * LAYERS[layerIndex + 1] + toIndex
          const tPosition = 0.22 + ((indexInLayer + 0.5) / totalInLayer) * 0.56
          const curve = this.makeCurve(from, to, seed)
          const point = curve.getPoint(tPosition)
          const tangent = curve.getTangent(tPosition)
          const perpendicular = new THREE.Vector3(-tangent.y, tangent.x, 0).normalize()
          const side = (fromIndex + toIndex) % 2 === 0 ? 1 : -1
          const offsetDistance = 0.5 + indexInLayer * 0.15

          point.addScaledVector(perpendicular, side * offsetDistance)
          point.z += ((indexInLayer % 3) - 1) * 0.4

          weightLabelData.push({ position: point, weight })
        }
      }
    }

    const minimumDistance = 1.8

    for (let iteration = 0; iteration < 60; iteration += 1) {
      for (let leftIndex = 0; leftIndex < weightLabelData.length; leftIndex += 1) {
        for (
          let rightIndex = leftIndex + 1;
          rightIndex < weightLabelData.length;
          rightIndex += 1
        ) {
          const left = weightLabelData[leftIndex].position
          const right = weightLabelData[rightIndex].position
          const dx = right.x - left.x
          const dy = right.y - left.y
          const dz = right.z - left.z
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

          if (distance < minimumDistance && distance > 0.001) {
            const push = (minimumDistance - distance) * 0.5
            const normalX = dx / distance
            const normalY = dy / distance
            const normalZ = dz / distance

            left.x -= normalX * push * 0.3
            left.y -= normalY * push
            left.z -= normalZ * push
            right.x += normalX * push * 0.3
            right.y += normalY * push
            right.z += normalZ * push
          }
        }
      }
    }

    weightLabelData.forEach((labelData) => {
      const spriteRecord = this.createWeightSprite(labelData.weight)
      spriteRecord.sprite.position.set(
        labelData.position.x,
        labelData.position.y,
        labelData.position.z,
      )
      this.networkGroup.add(spriteRecord.sprite)
      this.weightSprites.push(spriteRecord)
    })

    for (let layerIndex = 0; layerIndex < LAYERS.length; layerIndex += 1) {
      const spriteRecord = this.createLayerLabelSprite(
        LAYER_NAMES[layerIndex],
        LAYER_FUNCTIONS[layerIndex],
      )
      const x = layerIndex * LAYER_SPACING - ((LAYERS.length - 1) * LAYER_SPACING) / 2

        spriteRecord.sprite.position.set(
          x,
          ((LAYERS[layerIndex] - 1) / 2) * NODE_SPACING + 1.35,
          0,
        )

      this.networkGroup.add(spriteRecord.sprite)
      this.layerSprites.push(spriteRecord)
    }
  }

  createWeightSprite(weight) {
    const texture = buildWeightTexture(weight)
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.92,
      depthTest: false,
    })
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(1.8, 0.52, 1)
    sprite.renderOrder = 999

    return { sprite, material, texture }
  }

  createLayerLabelSprite(name, activationFn) {
    const texture = buildLayerTexture(name, activationFn)
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 1,
      depthTest: false,
    })
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(5.2, 1.28, 1)
    sprite.renderOrder = 999

    return { sprite, material, texture }
  }

  createMiniLabelSprite(text) {
    const texture = buildMiniLabelTexture(text)
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.94,
      depthTest: false,
    })
    const sprite = new THREE.Sprite(material)
    sprite.scale.set(2.2, 0.7, 1)
    sprite.renderOrder = 999

    return { sprite, material, texture }
  }

  refreshWeightSprite(spriteRecord, weight) {
    const texture = buildWeightTexture(weight)

    spriteRecord.texture.dispose()
    spriteRecord.texture = texture
    spriteRecord.material.map = texture
    spriteRecord.material.needsUpdate = true
  }

  createDecisionSurface() {
    this.decisionSurface = null
  }

  updateDecisionSurface() {
    return
  }

  makeCurve(from, to, seed) {
    const mid = from.clone().lerp(to, 0.5)
    mid.z += Math.sin(seed * 0.731) * 1.8
    mid.y += Math.cos(seed * 0.513) * 0.6

    return new THREE.QuadraticBezierCurve3(from.clone(), mid, to.clone())
  }

  makeBaseTube(curve, weight) {
    const absoluteWeight = Math.abs(weight)
    const radius = 0.012 + absoluteWeight * 0.1
    const baseColor = getWeightColor3(weight)
    const baseOpacity = 0.18 + absoluteWeight * 0.28
    const geometry = new THREE.TubeGeometry(curve, 48, radius, 5, false)
    const material = new THREE.MeshBasicMaterial({
      color: baseColor,
      transparent: true,
      opacity: baseOpacity,
    })

    return {
      mesh: new THREE.Mesh(geometry, material),
      geometry,
      material,
      radius,
      baseColor: baseColor.clone(),
      baseOpacity,
    }
  }

  flashNeuron(layerIndex, neuronIndex, kind, intensity) {
    const neuron = this.neuronMap[layerIndex]?.[neuronIndex]

    if (!neuron) {
      return
    }

    if (kind === 'forward') {
      neuron.forwardFlash = Math.max(neuron.forwardFlash, intensity)
      return
    }

    neuron.backwardFlash = Math.max(neuron.backwardFlash, intensity)
  }

  createTrainingBurst(connection, kind, direction, delay, intensity) {
    if (!this.networkGroup) {
      return
    }

    const geometry = new THREE.CylinderGeometry(
      0.055 + intensity * 0.11,
      0.04 + intensity * 0.09,
      0.5,
      8,
      1,
    )
    const material = new THREE.MeshBasicMaterial({
      color: kind === 'forward' ? COLORS.forward : COLORS.backward,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.visible = false
    this.networkGroup.add(mesh)

    this.trainingBursts.push({
      mesh,
      material,
      connection,
      kind,
      direction,
      delay,
      progress: direction > 0 ? 0 : 1,
      speed: 0.016 + intensity * 0.032,
      intensity,
      impacted: false,
    })
  }

  disposeTrainingBurst(burst) {
    burst.mesh.parent?.remove(burst.mesh)
    burst.mesh.geometry.dispose()
    burst.material.dispose()
  }

  clearTrainingBursts() {
    this.trainingBursts.forEach((burst) => {
      this.disposeTrainingBurst(burst)
    })
    this.trainingBursts = []
  }

  playSnapshotCue(snapshot) {
    if (!snapshot?.meta) {
      return
    }

    const { meta } = snapshot
    const outputLayerIndex = LAYERS.length - 1

    if (this.trainingBursts.length > 120) {
      const staleBursts = this.trainingBursts.splice(0, this.trainingBursts.length - 120)
      staleBursts.forEach((burst) => {
        this.disposeTrainingBurst(burst)
      })
    }

    this.trainingCueText = 'Forward batch + gradientes Excel + actualizacion de pesos'

    meta.input.forEach((value, index) => {
      this.flashNeuron(0, index, 'forward', 0.15 + value * 0.85)
    })

    meta.gradients[outputLayerIndex].forEach((value, index) => {
      this.flashNeuron(outputLayerIndex, index, 'backward', 0.18 + Math.abs(value) * 6)
    })

    this.connections.forEach((connection) => {
      const forwardActivation = meta.forward[connection.layer]?.[connection.fromIndex] ?? 0
      const backwardSignal = Math.abs(
        meta.gradients[connection.layer + 1]?.[connection.toIndex] ?? 0,
      )
      const weightShift = Math.abs(
        meta.connectionUpdates[connection.layer]?.[connection.fromIndex]?.[connection.toIndex] ?? 0,
      )
      const forwardIntensity = clamp(
        0.16 + forwardActivation * 0.72 + Math.abs(connection.weight) * 0.22,
        0.12,
        1,
      )
      const backwardIntensity = clamp(
        0.14 + backwardSignal * 5 + weightShift * 4.5,
        0.1,
        1,
      )

      this.createTrainingBurst(
        connection,
        'forward',
        1,
        connection.layer * 0.12 + ((connection.fromIndex + connection.toIndex) % 3) * 0.015,
        forwardIntensity,
      )
      this.createTrainingBurst(
        connection,
        'backward',
        -1,
        0.34 +
          (LAYERS.length - 2 - connection.layer) * 0.14 +
          ((connection.fromIndex * 2 + connection.toIndex) % 3) * 0.018,
        backwardIntensity,
      )
      connection.forwardFlash = Math.max(connection.forwardFlash, forwardIntensity * 0.25)
      connection.backwardFlash = Math.max(connection.backwardFlash, backwardIntensity * 0.25)
    })
  }

  attachListeners() {
    this.registerListener(document, 'keydown', this.handleKeyDown)
    this.registerListener(document, 'keyup', this.handleKeyUp)
    this.registerListener(document, 'pointerlockchange', this.handlePointerLockChange)
    this.registerListener(document, 'mousemove', this.handleMouseMove)
    this.registerListener(window, 'resize', this.handleResize)
    this.registerListener(this.renderer.domElement, 'click', this.handleCanvasClick)
    this.registerListener(this.renderer.domElement, 'wheel', this.handleWheel, {
      passive: true,
    })
  }

  registerListener(target, type, handler, options) {
    target.addEventListener(type, handler, options)
    this.listeners.push({ target, type, handler, options })
  }

  removeListeners() {
    this.listeners.forEach(({ target, type, handler, options }) => {
      target.removeEventListener(type, handler, options)
    })
    this.listeners = []
  }

  aimCameraAt(targetX, targetY, targetZ) {
    const dx = targetX - this.camX
    const dy = targetY - this.camY
    const dz = targetZ - this.camZ
    const distance = Math.max(0.001, Math.sqrt(dx * dx + dz * dz))

    this.yaw = Math.atan2(dx, -dz)
    this.pitch = Math.atan2(dy, distance)
  }

  handleResize() {
    if (!this.camera || !this.renderer) {
      return
    }

    this.camera.aspect = window.innerWidth / window.innerHeight
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(window.innerWidth, window.innerHeight)
    this.composer?.setSize(window.innerWidth, window.innerHeight)
    this.bloomPass?.setSize?.(window.innerWidth, window.innerHeight)
  }

  handleKeyDown(event) {
    if (this.tourActive) {
      return
    }

    if (event.code === 'ArrowRight') {
      event.preventDefault()
      this.stepForwardNav()
      return
    }

    if (event.code === 'ArrowLeft') {
      event.preventDefault()
      this.stepBack()
      return
    }

    if (event.code === 'Space' && !this.pointerLocked) {
      event.preventDefault()
      this.toggleAutoTrain()
      return
    }

    this.keys[event.code] = true

    if (event.code === 'ArrowUp' || event.code === 'ArrowDown') {
      event.preventDefault()
    }
  }

  handleKeyUp(event) {
    this.keys[event.code] = false
  }

  handleCanvasClick() {
    if (!this.renderer || this.tourActive) {
      return
    }

    if (!this.pointerLocked) {
      this.renderer.domElement.requestPointerLock?.()
      return
    }

    const rect = this.renderer.domElement.getBoundingClientRect()

    this.handleNeuronPick(rect.left + rect.width / 2, rect.top + rect.height / 2)
  }

  handlePointerLockChange() {
    this.pointerLocked = document.pointerLockElement === this.renderer?.domElement
    document.body.style.cursor = this.pointerLocked ? 'none' : 'default'
    this.emitState()
  }

  handleMouseMove(event) {
    if (!this.pointerLocked || this.tourActive) {
      return
    }

    this.yaw += event.movementX * 0.0022
    this.pitch -= event.movementY * 0.0022
    this.pitch = clamp(this.pitch, -1.35, 1.35)
  }

  handleWheel(event) {
    if (this.tourActive) {
      return
    }

    const speed = event.deltaY * 0.04
    const { width, depth } = ROOM_DIMENSIONS

    this.camX += Math.sin(this.yaw) * speed
    this.camZ -= Math.cos(this.yaw) * speed
    this.camX = clamp(this.camX, -width / 2 + 1, width / 2 - 1)
    this.camZ = clamp(this.camZ, -depth / 2 + 1, depth / 2 - 1)
  }

  applyMovement() {
    if (this.tourActive) {
      return
    }

    const { width, height, depth } = ROOM_DIMENSIONS
    const forwardX = Math.sin(this.yaw)
    const forwardZ = -Math.cos(this.yaw)
    const rightX = Math.cos(this.yaw)
    const rightZ = Math.sin(this.yaw)

    if (this.keys.KeyW || this.keys.ArrowUp) {
      this.camX += forwardX * this.speed
      this.camZ += forwardZ * this.speed
    }

    if (this.keys.KeyS || this.keys.ArrowDown) {
      this.camX -= forwardX * this.speed
      this.camZ -= forwardZ * this.speed
    }

    if (this.keys.KeyA) {
      this.camX -= rightX * this.speed
      this.camZ -= rightZ * this.speed
    }

    if (this.keys.KeyD) {
      this.camX += rightX * this.speed
      this.camZ += rightZ * this.speed
    }

    if (this.keys.Space) {
      this.camY += this.speed
    }

    if (this.keys.ShiftLeft || this.keys.ShiftRight) {
      this.camY -= this.speed
    }

    this.camX = clamp(this.camX, -width / 2 + 1, width / 2 - 1)
    this.camY = clamp(this.camY, -height / 2 + 1, height / 2 - 1)
    this.camZ = clamp(this.camZ, -depth / 2 + 1, depth / 2 - 1)
  }

  enterScene() {
    this.startGuidedTour()
  }

  toggleAutoTrain() {
    if (this.tourActive) {
      return
    }

    this.autoTrain = !this.autoTrain
    this.statusMode = this.autoTrain ? 'auto' : 'paused'
    this.emitState()
  }

  resetTraining() {
    if (this.tourActive) {
      return
    }

    this.autoTrain = false
    this.statusMode = 'ready'
    this.clearTrainingBursts()
    this.clearSelection()
    this.engine.initialize()
    this.applyCurrentModelToScene()
    this.emitState()
  }

  stepForward() {
    if (this.tourActive) {
      return
    }

    const snapshot = this.engine.stepForward()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.playSnapshotCue(snapshot)
    this.emitState()
  }

  stepBack() {
    if (this.tourActive) {
      return
    }

    const snapshot = this.engine.stepBack()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.playSnapshotCue(snapshot)
    this.emitState()
  }

  stepForwardNav() {
    if (this.tourActive) {
      return
    }

    const snapshot = this.engine.stepForwardNav()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.playSnapshotCue(snapshot)
    this.emitState()
  }

  stepEpochForward() {
    if (this.tourActive) {
      return
    }

    const snapshot = this.engine.stepEpochForward()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.playSnapshotCue(snapshot)
    this.emitState()
  }

  syncStatusMode() {
    const snapshot = this.engine.getCurrentSnapshot()

    if (!snapshot) {
      this.statusMode = 'ready'
      return
    }

    if (this.autoTrain) {
      this.statusMode = 'auto'
      return
    }

    if (this.statusMode === 'converged' || this.statusMode === 'auto') {
      this.statusMode = snapshot.step === 0 ? 'ready' : 'paused'
    }
  }

  applyCurrentModelToScene() {
    this.rebuildConnections()
    this.updateDecisionSurface()

    this.neurons.forEach((neuron) => {
      neuron.activation = this.engine.activations[neuron.layer][neuron.index]
      const brightness = 0.22 + neuron.activation * 0.4

      neuron.material.color.setHSL(0.58, 0.75, brightness)
      neuron.material.emissive.setHSL(0.58, 0.9, 0.07 + neuron.activation * 0.22)
      neuron.light.intensity = 0.6 + neuron.activation * 0.9
      neuron.light.color.set(COLORS.nodeLight)
    })

    if (this.selectedNeuron) {
      this.applySelectionHighlight(this.selectedNeuron)
    }

  }

  rebuildConnections() {
    let connectionIndex = 0

    for (let layerIndex = 0; layerIndex < LAYERS.length - 1; layerIndex += 1) {
      for (let fromIndex = 0; fromIndex < LAYERS[layerIndex]; fromIndex += 1) {
        for (let toIndex = 0; toIndex < LAYERS[layerIndex + 1]; toIndex += 1) {
          const connection = this.connections[connectionIndex]
          const weight = this.engine.weights[layerIndex][fromIndex][toIndex]
          const absoluteWeight = Math.abs(weight)
          const color = getWeightColor3(weight)
          const pulseRadius = 0.025 + absoluteWeight * 0.13

          connection.weight = weight
          connection.base.baseColor.copy(color)
          connection.base.baseOpacity = 0.18 + absoluteWeight * 0.28
          connection.base.material.color.copy(color)
          connection.base.material.opacity = connection.base.baseOpacity

          connection.pulses.forEach((pulse) => {
            pulse.radius = pulseRadius
            pulse.material.color.copy(color)
          })

          this.refreshWeightSprite(this.weightSprites[connectionIndex], weight)
          connectionIndex += 1
        }
      }
    }
  }

  handleNeuronPick(clientX, clientY) {
    if (!this.renderer || !this.camera) {
      return
    }

    const bounds = this.renderer.domElement.getBoundingClientRect()
    const normalizedX = this.pointerLocked
      ? 0
      : ((clientX - bounds.left) / bounds.width) * 2 - 1
    const normalizedY = this.pointerLocked
      ? 0
      : -(((clientY - bounds.top) / bounds.height) * 2 - 1)

    this.raycaster.setFromCamera(new THREE.Vector2(normalizedX, normalizedY), this.camera)

    const hits = this.raycaster.intersectObjects(this.neurons.map((neuron) => neuron.mesh))

    if (hits.length > 0) {
      const hitNeuron = this.neurons.find((neuron) => neuron.mesh === hits[0].object)

      if (hitNeuron) {
        this.selectNeuron(hitNeuron)
      }
    } else {
      this.clearSelection()
    }

    this.emitState()
  }

  selectNeuron(neuron) {
    if (this.selectedNeuron && this.selectedNeuron !== neuron) {
      this.removeSelectionHighlight(this.selectedNeuron)
    }

    this.selectedNeuron = neuron
    neuron.selected = true
    this.applySelectionHighlight(neuron)
  }

  clearSelection() {
    if (!this.selectedNeuron) {
      return
    }

    this.removeSelectionHighlight(this.selectedNeuron)
    this.selectedNeuron = null
  }

  applySelectionHighlight(neuron) {
    neuron.selected = true
    neuron.material.emissive.setHSL(0.58, 1, 0.4)
    neuron.light.intensity = 3
    neuron.light.color.set(COLORS.selected)
  }

  removeSelectionHighlight(neuron) {
    neuron.selected = false
    neuron.material.emissive.setHSL(0.58, 0.9, 0.07 + neuron.activation * 0.22)
    neuron.light.intensity = 0.6 + neuron.activation * 0.9
    neuron.light.color.set(COLORS.nodeLight)
  }

  buildSelectedNeuronData() {
    if (!this.selectedNeuron) {
      return null
    }

    const { layer, index } = this.selectedNeuron
    const activation = this.engine.activations[layer][index]
    const incomingWeights = this.engine.getIncomingWeights(layer, index).map(
      (weight, sourceIndex) => ({
        label: `n${sourceIndex + 1} → ${weight >= 0 ? '+' : ''}${weight.toFixed(4)}`,
        percent: Number(Math.min(100, Math.abs(weight) * 100).toFixed(0)),
        color: getWeightColorCss(weight),
      }),
    )

    return {
      title: `${LAYER_NAMES[layer]} · N${index + 1}`,
      layerName: LAYER_NAMES[layer],
      index: index + 1,
      activation: activation.toFixed(4),
      activationTone: getActivationTone(activation),
      functionName: LAYER_FUNCTIONS[layer],
      incomingWeights,
      isInputLayer: layer === 0,
    }
  }

  buildViewState() {
    const snapshot = this.engine.getCurrentSnapshot()
    const status = getStatusMeta(this.statusMode)
    const currentSampleIndex = snapshot?.sampleIndex ?? 0
    const outputs = this.engine.getDatasetPredictions().map((prediction, index) => ({
      label: formatSampleLabel(prediction.input, prediction.target),
      value: prediction.prediction.toFixed(2),
      correct: isPredictionAligned(prediction.prediction, prediction.target),
      active: index === currentSampleIndex,
    }))

    return {
      architectureLabel: ARCHITECTURE_LABEL,
      connectionCount: TOTAL_CONNECTIONS,
      epoch: snapshot?.step ?? 0,
      entryVisible: this.entryVisible,
      mode3D: true,
      pointerLocked: this.pointerLocked,
      cinematicActive: this.tourActive,
      mouseLabel: this.tourActive
        ? 'Room tour automatico en curso'
        : 'Clic para explorar · WASD · ESC para soltar',
      infoVisible: Boolean(this.selectedNeuron),
      selectedNeuron: this.buildSelectedNeuronData(),
      training: {
        step: snapshot?.step ?? 0,
        lossText: snapshot ? snapshot.loss.toFixed(4) : '—',
        lossTone: snapshot
          ? snapshot.loss < 0.05
            ? 'good'
            : snapshot.loss < 0.2
              ? 'ok'
              : 'bad'
          : '',
        lrText: String(LEARNING_RATE),
        statusText: status.text,
        statusTone: status.tone,
        stepPosition: snapshot
          ? `muestra ${snapshot.sampleIndex + 1}/${snapshot.sampleCount} · frame ${this.engine.historyIndex + 1}/${this.engine.history.length}`
          : '0 / 0',
        activeSampleText: snapshot
          ? `muestra activa ${snapshot.sampleIndex + 1} de ${snapshot.sampleCount}`
          : 'muestra activa —',
        sampleText: snapshot
          ? `epoca ${snapshot.step} · ${formatSampleLabel(snapshot.inp, snapshot.tgt)}`
          : '—',
        visualText: this.tourActive ? this.tourLabel : this.trainingCueText,
        outputs: outputs.length > 0 ? outputs : createPlaceholderOutputs(),
        losses:
          this.engine.generatedEpochs.length > 0
            ? this.engine.generatedEpochs.map((epochTrace) => epochTrace.averageLoss)
            : [0],
        currentLossIndex: snapshot ? Math.max(snapshot.step - 1, 0) : 0,
      },
    }
  }

  emitState() {
    this.onStateChange?.(this.buildViewState())
  }

  updateCamera(delta) {
    if (this.tourActive) {
      this.advanceGuidedTour(delta)
      return
    }

    this.camera.position.set(this.camX, this.camY, this.camZ)

    const directionX = Math.sin(this.yaw) * Math.cos(this.pitch)
    const directionY = Math.sin(this.pitch)
    const directionZ = -Math.cos(this.yaw) * Math.cos(this.pitch)

    this.camera.lookAt(
      this.camX + directionX,
      this.camY + directionY,
      this.camZ + directionZ,
    )
  }

  animateNodes() {
    this.neurons.forEach((neuron) => {
      neuron.forwardFlash *= 0.92
      neuron.backwardFlash *= 0.9

      if (!neuron.selected) {
        neuron.material.emissive.setHSL(
          0.58,
          0.9,
          0.07 +
            neuron.activation * 0.22 +
            Math.sin(this.tick * 1.2 + neuron.index * 0.6) * 0.03,
        )
        neuron.material.color.setHSL(
          0.58,
          0.75,
          0.22 + neuron.activation * 0.4 + neuron.forwardFlash * 0.08,
        )
        neuron.material.emissive.lerp(
          this.forwardSignalColor,
          Math.min(0.65, neuron.forwardFlash * 0.55),
        )
        neuron.material.emissive.lerp(
          this.backwardSignalColor,
          Math.min(0.72, neuron.backwardFlash * 0.58),
        )
        neuron.light.intensity =
          0.6 +
          neuron.activation * 0.9 +
          Math.sin(this.tick * 1.5 + neuron.index) * 0.12 +
          neuron.forwardFlash * 1.8 +
          neuron.backwardFlash * 2.2
        neuron.light.color.set(COLORS.nodeLight)
        neuron.light.color.lerp(
          this.forwardSignalColor,
          Math.min(0.7, neuron.forwardFlash * 0.5),
        )
        neuron.light.color.lerp(
          this.backwardSignalColor,
          Math.min(0.8, neuron.backwardFlash * 0.58),
        )
      }

      neuron.mesh.scale.setScalar(
        1 +
          Math.sin(this.tick + neuron.index * 0.7) * 0.022 +
          neuron.forwardFlash * 0.06 +
          neuron.backwardFlash * 0.075,
      )
    })
  }

  getPulseSpeed(connection) {
    const sourceActivation = this.engine.activations[connection.layer][connection.fromIndex]
    return 0.003 + sourceActivation * 0.006 + Math.abs(connection.weight) * 0.003
  }

  animatePulses() {
    this.connections.forEach((connection) => {
      const speed = this.getPulseSpeed(connection)
      const absoluteWeight = Math.abs(connection.weight)

      connection.pulses.forEach((pulse) => {
        pulse.t = (pulse.t + speed) % 1

        const halfLength = 0.028
        const startT = Math.max(0.001, pulse.t - halfLength)
        const endT = Math.min(0.999, pulse.t + halfLength)
        const positionA = connection.curve.getPoint(startT)
        const positionB = connection.curve.getPoint(endT)

        pulse.mesh.position.set(
          (positionA.x + positionB.x) / 2,
          (positionA.y + positionB.y) / 2,
          (positionA.z + positionB.z) / 2,
        )

        this.tempTangent.subVectors(positionB, positionA).normalize()
        this.tempQuaternion.setFromUnitVectors(this.tempUp, this.tempTangent)
        pulse.mesh.quaternion.copy(this.tempQuaternion)

        const segmentLength = positionA.distanceTo(positionB)
        pulse.mesh.scale.set(pulse.radius, segmentLength, pulse.radius)
        pulse.material.opacity = (0.65 + absoluteWeight * 0.3) * Math.sin(pulse.t * Math.PI)
      })
    })
  }

  animateConnectionBases() {
    this.connections.forEach((connection) => {
      connection.forwardFlash *= 0.92
      connection.backwardFlash *= 0.9

      connection.base.material.color.copy(connection.base.baseColor)
      connection.base.material.color.lerp(
        this.forwardSignalColor,
        Math.min(0.7, connection.forwardFlash * 0.5),
      )
      connection.base.material.color.lerp(
        this.backwardSignalColor,
        Math.min(0.8, connection.backwardFlash * 0.6),
      )
      connection.base.material.opacity = clamp(
        connection.base.baseOpacity +
          connection.forwardFlash * 0.2 +
          connection.backwardFlash * 0.22,
        0.1,
        0.96,
      )
    })
  }

  animateTrainingBursts() {
    if (this.trainingBursts.length === 0) {
      return
    }

    this.trainingBursts = this.trainingBursts.filter((burst) => {
      if (burst.delay > 0) {
        burst.delay -= 0.016
        return true
      }

      burst.mesh.visible = true
      burst.progress += burst.speed * burst.direction
      burst.connection[burst.kind === 'forward' ? 'forwardFlash' : 'backwardFlash'] = Math.max(
        burst.connection[burst.kind === 'forward' ? 'forwardFlash' : 'backwardFlash'],
        burst.intensity * 0.42,
      )

      const endThreshold = burst.direction > 0 ? 0.9 : 0.1

      if (
        !burst.impacted &&
        ((burst.direction > 0 && burst.progress >= endThreshold) ||
          (burst.direction < 0 && burst.progress <= endThreshold))
      ) {
        const layerIndex = burst.direction > 0 ? burst.connection.layer + 1 : burst.connection.layer
        const neuronIndex = burst.direction > 0 ? burst.connection.toIndex : burst.connection.fromIndex

        this.flashNeuron(layerIndex, neuronIndex, burst.kind, burst.intensity)
        burst.impacted = true
      }

      if (burst.progress <= 0 || burst.progress >= 1) {
        this.disposeTrainingBurst(burst)
        return false
      }

      const halfLength = 0.04 + burst.intensity * 0.02
      const startT = clamp(burst.progress - halfLength, 0.001, 0.999)
      const endT = clamp(burst.progress + halfLength, 0.001, 0.999)
      const positionA = burst.connection.curve.getPoint(startT)
      const positionB = burst.connection.curve.getPoint(endT)
      const fade = Math.sin(Math.min(1, burst.progress) * Math.PI)

      burst.mesh.position.set(
        (positionA.x + positionB.x) / 2,
        (positionA.y + positionB.y) / 2,
        (positionA.z + positionB.z) / 2,
      )

      this.tempTangent.subVectors(positionB, positionA).normalize()
      this.tempQuaternion.setFromUnitVectors(this.tempUp, this.tempTangent)
      burst.mesh.quaternion.copy(this.tempQuaternion)
      burst.mesh.scale.set(1, positionA.distanceTo(positionB) / 0.5, 1)
      burst.material.opacity = (0.32 + burst.intensity * 0.58) * fade

      return true
    })
  }

  animateDecisionSurface() {
    if (!this.decisionSurface) {
      return
    }

    const energy = Math.min(1, this.trainingBursts.length / 18)

    this.decisionSurface.group.position.y = -0.04 + Math.sin(this.tick * 0.45) * 0.1
    this.decisionSurface.group.rotation.y = Math.sin(this.tick * 0.2) * 0.04
    this.decisionSurface.beam.material.opacity = 0.055 + energy * 0.045
    this.decisionSurface.plane.material.opacity = 0.18 + energy * 0.04
    this.decisionSurface.frame.material.opacity = 0.12 + energy * 0.05
    this.decisionSurface.glow.material.opacity = 0.045 + energy * 0.035
    this.decisionSurface.baseGlow.material.opacity = 0.055 + energy * 0.045
    this.decisionSurface.markers.forEach((marker) => {
      marker.ring.position.y = -6.04 + Math.sin(this.tick * 1.6 + marker.phase) * 0.05
      marker.ring.material.opacity = 0.24 + Math.sin(this.tick * 1.8 + marker.phase) * 0.07
    })
  }

  animateLabEnvironment() {
    const energy = Math.min(1, this.trainingBursts.length / 18)
    const snapshot = this.engine.getCurrentSnapshot()
    const outputs = this.engine.getDatasetPredictions().map((prediction) => ({
      label: formatSampleLabel(prediction.input, prediction.target),
      value: prediction.prediction.toFixed(2),
      correct: isPredictionAligned(prediction.prediction, prediction.target),
    }))

    this.labAccentMaterials.forEach((accent) => {
      accent.material.emissiveIntensity =
        accent.baseIntensity +
        Math.sin(this.tick * accent.speed + accent.phase) * accent.pulseRange +
        energy * 0.08
    })

    this.labAccentLights.forEach((accent) => {
      accent.light.intensity =
        accent.baseIntensity +
        Math.sin(this.tick * accent.speed + accent.phase) * accent.pulseRange +
        energy * 0.22
    })

    this.labFans.forEach((fan, index) => {
      fan.mesh.rotation.x += fan.speed + energy * 0.2 + index * 0.002
    })

    this.labSteam.forEach((steam) => {
      steam.mesh.position.y = steam.baseY + Math.sin(this.tick * 1.4 + steam.phase) * 0.08
      steam.mesh.position.x += Math.sin(this.tick * 0.5 + steam.phase) * 0.0015
      steam.mesh.material.opacity = 0.04 + Math.abs(Math.sin(this.tick * 1.2 + steam.phase)) * 0.08
      const scale = 0.85 + Math.sin(this.tick * 1.1 + steam.phase) * 0.12
      steam.mesh.scale.setScalar(scale)
    })

    this.labCoreRings.forEach((ring) => {
      ring.mesh.position.y += Math.sin(this.tick * 0.9 + ring.phase) * 0.0015

      if (ring.axis === 'y') {
        ring.mesh.rotation.y += ring.speed * 0.01 + energy * 0.006
      } else {
        ring.mesh.rotation.z += ring.speed * 0.01 + energy * 0.008
      }

      ring.mesh.material.emissiveIntensity = 0.14 + Math.sin(this.tick * 1.2 + ring.phase) * 0.04 + energy * 0.12
    })

    this.labBots.forEach((bot) => {
      if (bot.kind === 'robot') {
        bot.body.position.y = bot.anchor.y + Math.sin(this.tick * 1.2 + bot.phase) * 0.09
        bot.head.position.y = bot.body.position.y + 0.48
        bot.head.rotation.y = Math.sin(this.tick * 0.9 + bot.phase) * 0.3
      }

      if (bot.kind === 'arm') {
        bot.body.rotation.z = Math.sin(this.tick * 0.8 + bot.phase) * 0.12
        bot.head.rotation.z = 0.4 + Math.sin(this.tick * 1.1 + bot.phase) * 0.32
        bot.probe.position.x = bot.anchor.x - 1.1 + Math.sin(this.tick * 0.7 + bot.phase) * 0.3
        bot.probe.position.y = bot.anchor.y + 1.9 + Math.cos(this.tick * 0.9 + bot.phase) * 0.35
      }

      if (bot.kind === 'roomba') {
        const x = bot.anchor.x + Math.sin(this.tick * 0.42 + bot.phase) * 1.8
        const z = bot.anchor.z + Math.cos(this.tick * 0.34 + bot.phase) * 1.25
        const yaw = Math.sin(this.tick * 0.38 + bot.phase) * 0.9

        bot.body.position.set(x, bot.anchor.y, z)
        bot.body.rotation.y = yaw

        bot.top.position.set(x, bot.anchor.y + 0.16, z)
        bot.top.rotation.y = yaw + this.tick * 1.4

        bot.head.position.set(x, bot.anchor.y - 0.15, z)
        bot.head.rotation.y = yaw

        const forwardX = Math.sin(yaw)
        const forwardZ = Math.cos(yaw)

        bot.eyeLeft.position.set(x - forwardZ * 0.16 + forwardX * 0.3, bot.anchor.y + 0.04, z + forwardX * 0.16 + forwardZ * 0.3)
        bot.eyeRight.position.set(x + forwardZ * 0.16 + forwardX * 0.3, bot.anchor.y + 0.04, z - forwardX * 0.16 + forwardZ * 0.3)
      }
    })

    if (this.heroSpotlight) {
      this.heroSpotlight.intensity = 1.18 + energy * 0.42 + Math.sin(this.tick * 0.9) * 0.08
      this.heroSpotlight.angle = 0.4 + Math.sin(this.tick * 0.25) * 0.02
    }

    const panelRefreshInterval = this.tourActive ? 10 : 6

    if (this.frameCount % panelRefreshInterval === 0) {
      this.displayPanels.forEach((panel, index) => {
        renderDisplayPanelTexture(panel, {
          tick: this.tick + index * 0.45,
          loss: snapshot?.loss ?? 0.5,
          epoch: snapshot?.step ?? 0,
          outputs,
          activity: energy + (panel.mode === 'hero' ? 0.14 : 0),
        })
      })
    }

  }

  animate(now = performance.now()) {
    this.animationFrameId = requestAnimationFrame(this.animate)
    if (!this.lastFrameTime) {
      this.lastFrameTime = now
    }

    const delta = clamp((now - this.lastFrameTime) / 1000, 1 / 240, 0.05)
    this.lastFrameTime = now

    this.tick += delta * 0.72
    this.frameCount += 1

    this.applyMovement()
    this.updateCamera(delta)
    this.animateNodes()
    this.animatePulses()
    this.animateConnectionBases()
    this.animateTrainingBursts()
    this.animateDecisionSurface()
    this.animateLabEnvironment()

    if (this.bloomPass) {
      const tourBloomCap = this.tourActive ? 0.07 : 0.12
      const targetStrength = 0.2 + Math.min(tourBloomCap, this.trainingBursts.length * 0.006)
      this.bloomPass.strength = lerp(this.bloomPass.strength, targetStrength, 0.08)
    }

    if (this.autoTrain && this.frameCount % 20 === 0) {
      this.stepForward()
    }

    if (this.composer) {
      this.composer.render()
      return
    }

    this.renderer.render(this.scene, this.camera)
  }

  disposeScene() {
    this.clearTrainingBursts()

    if (this.scene) {
      this.scene.traverse((object) => {
        if (object.geometry) {
          object.geometry.dispose?.()
        }

        if (object.material) {
          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material]

          materials.forEach((material) => {
            material.map?.dispose?.()
            material.dispose?.()
          })
        }
      })
    }

    this.weightSprites.forEach((spriteRecord) => {
      spriteRecord.texture.dispose?.()
      spriteRecord.material.dispose?.()
    })

    this.layerSprites.forEach((spriteRecord) => {
      spriteRecord.texture.dispose?.()
      spriteRecord.material.dispose?.()
    })

    this.renderer?.dispose?.()
    this.composer?.dispose?.()
    this.sharedPulseGeometry?.dispose?.()
    this.sharedPulseGeometry = null

    if (this.renderer?.domElement?.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement)
    }

    this.scene = null
    this.camera = null
    this.renderer = null
    this.composer = null
    this.renderPass = null
    this.bloomPass = null
    this.environmentGroup = null
    this.networkGroup = null
    this.neurons = []
    this.neuronMap = []
    this.connections = []
    this.weightSprites = []
    this.layerSprites = []
    this.decisionSurface = null
    this.labAccentMaterials = []
    this.labAccentLights = []
    this.labFans = []
    this.labSteam = []
    this.labBots = []
    this.labCoreRings = []
    this.displayPanels = []
    this.heroSpotlight = null
    this.selectedNeuron = null
  }
}
