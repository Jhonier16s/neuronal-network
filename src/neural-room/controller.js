import * as THREE from 'three'
import {
  ARCHITECTURE_LABEL,
  CAMERA_DEFAULT,
  CAMERA_ENTRY,
  COLORS,
  DASH_COUNT,
  DASH_GAP,
  LAYERS,
  LAYER_FUNCTIONS,
  LAYER_NAMES,
  LAYER_SPACING,
  LEARNING_RATE,
  MOVE_SPEED,
  NODE_SPACING,
  ROOM_DIMENSIONS,
  TOTAL_CONNECTIONS,
  XOR_DATA,
} from './constants.js'
import { NeuralNetworkEngine } from './engine.js'

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
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

function formatXorLabel(input, target) {
  return `${input[0]},${input[1]} → ${target[0]}`
}

function createPlaceholderOutputs() {
  return XOR_DATA.map(([input, target]) => ({
    label: formatXorLabel(input, target),
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
      return { text: '¡Convergió!', tone: 'good' }
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
  context.shadowBlur = 10
  context.fillText(name.toUpperCase(), 160, 28)
  context.font = '300 13px IBM Plex Mono, monospace'
  context.fillStyle = '#4a6878'
  context.shadowBlur = 0
  context.fillText(activationFn, 160, 52)

  return new THREE.CanvasTexture(canvas)
}

export function createInitialViewState() {
  return {
    architectureLabel: ARCHITECTURE_LABEL,
    connectionCount: TOTAL_CONNECTIONS,
    epoch: 0,
    entryVisible: true,
    mode3D: true,
    pointerLocked: false,
    mouseLabel: 'Clic para capturar ratón · WASD · ESC',
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
    this.canvas2d = null
    this.ctx2d = null
    this.scene = null
    this.camera = null
    this.renderer = null
    this.networkGroup = null
    this.neurons = []
    this.nodePointMap = []
    this.connections = []
    this.weightSprites = []
    this.layerSprites = []
    this.selectedNeuron = null
    this.keys = {}
    this.listeners = []
    this.animationFrameId = null
    this.tick = 0
    this.frameCount = 0
    this.phase2d = 0
    this.camX = CAMERA_DEFAULT.x
    this.camY = CAMERA_DEFAULT.y
    this.camZ = CAMERA_DEFAULT.z
    this.yaw = 0
    this.pitch = 0
    this.speed = MOVE_SPEED
    this.raycaster = new THREE.Raycaster()
    this.tempUp = new THREE.Vector3(0, 1, 0)
    this.tempQuaternion = new THREE.Quaternion()
    this.tempTangent = new THREE.Vector3()

    this.animate = this.animate.bind(this)
    this.handleResize = this.handleResize.bind(this)
    this.handleKeyDown = this.handleKeyDown.bind(this)
    this.handleKeyUp = this.handleKeyUp.bind(this)
    this.handleCanvasClick = this.handleCanvasClick.bind(this)
    this.handlePointerLockChange = this.handlePointerLockChange.bind(this)
    this.handleMouseMove = this.handleMouseMove.bind(this)
    this.handleWheel = this.handleWheel.bind(this)
  }

  mount({ viewportEl, canvas2d }) {
    if (!viewportEl || !canvas2d) {
      return
    }

    this.viewportEl = viewportEl
    this.canvas2d = canvas2d
    this.ctx2d = canvas2d.getContext('2d')
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

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(COLORS.background)

    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      300,
    )
    this.camera.position.set(0, 6, 28)
    this.camera.lookAt(0, 0, 0)

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setSize(window.innerWidth, window.innerHeight)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1
    this.renderer.domElement.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;z-index:1;'
    this.viewportEl.appendChild(this.renderer.domElement)

    this.createRoom()
    this.networkGroup = new THREE.Group()
    this.scene.add(this.networkGroup)
    this.buildNetworkObjects()
  }

  createRoom() {
    const { width, height, depth } = ROOM_DIMENSIONS

    this.createWall(width, depth, COLORS.roomFloor, Math.PI / 2, 0, 0, 0, -height / 2, 0)
    this.createWall(width, depth, COLORS.roomCeiling, -Math.PI / 2, 0, 0, 0, height / 2, 0)
    this.createWall(width, height, COLORS.roomBack, 0, 0, 0, 0, 0, -depth / 2)
    this.createWall(width, height, COLORS.roomBack, 0, Math.PI, 0, 0, 0, depth / 2)
    this.createWall(depth, height, COLORS.roomSide, 0, Math.PI / 2, 0, -width / 2, 0, 0)
    this.createWall(depth, height, COLORS.roomSide, 0, -Math.PI / 2, 0, width / 2, 0, 0)

    const gridMaterial = new THREE.LineBasicMaterial({
      color: COLORS.grid,
      transparent: true,
      opacity: 0.7,
    })
    const gridStep = 3

    for (let gridX = -width / 2; gridX <= width / 2; gridX += gridStep) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(gridX, -height / 2, -depth / 2),
        new THREE.Vector3(gridX, -height / 2, depth / 2),
      ])

      this.scene.add(new THREE.Line(geometry, gridMaterial))
    }

    for (let gridZ = -depth / 2; gridZ <= depth / 2; gridZ += gridStep) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-width / 2, -height / 2, gridZ),
        new THREE.Vector3(width / 2, -height / 2, gridZ),
      ])

      this.scene.add(new THREE.Line(geometry, gridMaterial))
    }

    this.scene.add(new THREE.AmbientLight(COLORS.ambient, 7))

    const directionalLight = new THREE.DirectionalLight(COLORS.directional, 0.4)
    directionalLight.position.set(5, 12, 8)
    this.scene.add(directionalLight)

    ;[-12, 0, 12].forEach((positionX) => {
      const pointLight = new THREE.PointLight(COLORS.point, 2.8, 28)
      pointLight.position.set(positionX, height / 2 - 0.5, 0)
      this.scene.add(pointLight)
    })
  }

  createWall(width, height, color, rx, ry, rz, px, py, pz) {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      new THREE.MeshStandardMaterial({
        color,
        roughness: 1,
        side: THREE.DoubleSide,
      }),
    )

    mesh.rotation.set(rx, ry, rz)
    mesh.position.set(px, py, pz)
    this.scene.add(mesh)
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
    this.nodePointMap = []
    this.connections = []
    this.weightSprites = []
    this.layerSprites = []

    const sphereGeometry = new THREE.SphereGeometry(0.75, 40, 40)

    for (let layerIndex = 0; layerIndex < LAYERS.length; layerIndex += 1) {
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

        const pointLight = new THREE.PointLight(
          COLORS.nodeLight,
          0.7 + activation * 0.8,
          5,
        )
        pointLight.position.copy(position)
        this.networkGroup.add(pointLight)

        this.nodePointMap[layerIndex].push(position.clone())
        this.neurons.push({
          mesh,
          material,
          light: pointLight,
          layer: layerIndex,
          index: neuronIndex,
          activation,
          position: position.clone(),
          selected: false,
        })
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
          })
        }
      }
    }

    this.connections.forEach((connection, connectionIndex) => {
      const absoluteWeight = Math.abs(connection.weight)
      const radius = 0.025 + absoluteWeight * 0.13

      for (let dashIndex = 0; dashIndex < DASH_COUNT; dashIndex += 1) {
        const geometry = new THREE.CylinderGeometry(radius, radius, 0.5, 6, 1)
        const material = new THREE.MeshBasicMaterial({
          color: getWeightColor3(connection.weight),
          transparent: true,
          opacity: 0,
        })
        const mesh = new THREE.Mesh(geometry, material)

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
        ((LAYERS[layerIndex] - 1) / 2) * NODE_SPACING + 2.8,
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

  refreshWeightSprite(spriteRecord, weight) {
    const texture = buildWeightTexture(weight)

    spriteRecord.texture.dispose()
    spriteRecord.texture = texture
    spriteRecord.material.map = texture
    spriteRecord.material.needsUpdate = true
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
    const geometry = new THREE.TubeGeometry(curve, 48, radius, 5, false)
    const material = new THREE.MeshBasicMaterial({
      color: getWeightColor3(weight),
      transparent: true,
      opacity: 0.18 + absoluteWeight * 0.28,
    })

    return {
      mesh: new THREE.Mesh(geometry, material),
      geometry,
      material,
      radius,
    }
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

  handleResize() {
    if (!this.camera || !this.renderer) {
      return
    }

    this.camera.aspect = window.innerWidth / window.innerHeight
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(window.innerWidth, window.innerHeight)
    this.resize2DCanvas()

    if (!this.mode3D) {
      this.draw2D()
    }
  }

  resize2DCanvas() {
    if (!this.canvas2d) {
      return
    }

    this.canvas2d.width = window.innerWidth
    this.canvas2d.height = window.innerHeight
  }

  handleKeyDown(event) {
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
    if (!this.mode3D || !this.renderer) {
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
    if (!this.pointerLocked || !this.mode3D) {
      return
    }

    this.yaw += event.movementX * 0.0022
    this.pitch -= event.movementY * 0.0022
    this.pitch = clamp(this.pitch, -1.35, 1.35)
  }

  handleWheel(event) {
    if (!this.mode3D) {
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
    this.entryVisible = false
    this.mode3D = true
    this.camX = CAMERA_ENTRY.x
    this.camY = CAMERA_ENTRY.y
    this.camZ = CAMERA_ENTRY.z
    this.yaw = 0
    this.pitch = 0
    this.emitState()
  }

  toggleMode() {
    this.mode3D = !this.mode3D

    if (!this.mode3D && document.pointerLockElement === this.renderer?.domElement) {
      document.exitPointerLock?.()
    }

    if (!this.mode3D) {
      this.resize2DCanvas()
      this.draw2D()
    }

    this.emitState()
  }

  toggleAutoTrain() {
    this.autoTrain = !this.autoTrain
    this.statusMode = this.autoTrain ? 'auto' : 'paused'
    this.emitState()
  }

  stepForward() {
    this.engine.stepForward()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.emitState()
  }

  stepBack() {
    this.engine.stepBack()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.emitState()
  }

  stepForwardNav() {
    this.engine.stepForwardNav()
    this.syncStatusMode()
    this.applyCurrentModelToScene()
    this.emitState()
  }

  syncStatusMode() {
    const snapshot = this.engine.getCurrentSnapshot()

    if (!snapshot) {
      this.statusMode = 'ready'
      return
    }

    if (snapshot.loss < 0.01) {
      this.autoTrain = false
      this.statusMode = 'converged'
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

    if (!this.mode3D) {
      this.draw2D()
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
          const baseRadius = 0.012 + absoluteWeight * 0.1
          const pulseRadius = 0.025 + absoluteWeight * 0.13

          connection.weight = weight
          connection.base.material.color.copy(color)
          connection.base.material.opacity = 0.18 + absoluteWeight * 0.28
          connection.base.geometry.dispose()
          connection.base.geometry = new THREE.TubeGeometry(
            connection.curve,
            48,
            baseRadius,
            5,
            false,
          )
          connection.base.mesh.geometry = connection.base.geometry
          connection.base.radius = baseRadius

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
    const outputs = this.engine.getXorPredictions().map((prediction) => ({
      label: formatXorLabel(prediction.input, prediction.target),
      value: prediction.prediction.toFixed(2),
      correct: Math.abs(prediction.prediction - prediction.target[0]) < 0.5,
    }))

    return {
      architectureLabel: ARCHITECTURE_LABEL,
      connectionCount: TOTAL_CONNECTIONS,
      epoch: snapshot?.step ?? 0,
      entryVisible: this.entryVisible,
      mode3D: this.mode3D,
      pointerLocked: this.pointerLocked,
      mouseLabel: this.mode3D
        ? 'Clic para capturar ratón · WASD · ESC para soltar'
        : 'Vista 2D',
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
          ? `${this.engine.historyIndex + 1} / ${this.engine.history.length}`
          : '0 / 0',
        outputs: outputs.length > 0 ? outputs : createPlaceholderOutputs(),
        losses:
          this.engine.history.length > 0
            ? this.engine.history.map((historySnapshot) => historySnapshot.loss)
            : [0],
        currentLossIndex: Math.max(this.engine.historyIndex, 0),
      },
    }
  }

  emitState() {
    this.onStateChange?.(this.buildViewState())
  }

  updateCamera() {
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
      if (!neuron.selected) {
        neuron.material.emissive.setHSL(
          0.58,
          0.9,
          0.07 +
            neuron.activation * 0.22 +
            Math.sin(this.tick * 1.2 + neuron.index * 0.6) * 0.03,
        )
        neuron.light.intensity =
          0.6 +
          neuron.activation * 0.9 +
          Math.sin(this.tick * 1.5 + neuron.index) * 0.12
      }

      neuron.mesh.scale.setScalar(1 + Math.sin(this.tick + neuron.index * 0.7) * 0.022)
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
        pulse.mesh.scale.set(1, segmentLength / 0.5, 1)
        pulse.material.opacity = (0.65 + absoluteWeight * 0.3) * Math.sin(pulse.t * Math.PI)

        if (pulse.mesh.geometry.parameters.radiusTop !== pulse.radius) {
          pulse.mesh.geometry.dispose()
          pulse.mesh.geometry = new THREE.CylinderGeometry(
            pulse.radius,
            pulse.radius,
            0.5,
            6,
            1,
          )
        }
      })
    })
  }

  animate() {
    this.animationFrameId = requestAnimationFrame(this.animate)
    this.tick += 0.012
    this.frameCount += 1

    this.applyMovement()
    this.updateCamera()
    this.animateNodes()
    this.animatePulses()

    if (this.autoTrain && this.frameCount % 20 === 0) {
      this.stepForward()
    }

    if (!this.mode3D) {
      this.phase2d += 0.03
      this.draw2D()
      return
    }

    this.renderer.render(this.scene, this.camera)
  }

  get2DPositions() {
    const width = this.canvas2d.width
    const height = this.canvas2d.height
    const paddingX = width * 0.12
    const paddingY = height * 0.18
    const step = (width - paddingX * 2) / (LAYERS.length - 1)

    return LAYERS.map((count, layerIndex) => {
      return Array.from({ length: count }, (_, neuronIndex) => ({
        x: paddingX + layerIndex * step,
        y:
          height / 2 +
          (neuronIndex - (count - 1) / 2) * Math.min(88, (height - paddingY * 2) / (count + 1)),
      }))
    })
  }

  quadraticAt(p0, p1, p2, t) {
    return (1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2
  }

  draw2D() {
    if (!this.canvas2d || !this.ctx2d) {
      return
    }

    const width = this.canvas2d.width
    const height = this.canvas2d.height
    const context = this.ctx2d
    const positions = this.get2DPositions()

    context.fillStyle = '#06090d'
    context.fillRect(0, 0, width, height)
    context.strokeStyle = '#0c1620'
    context.lineWidth = 1

    for (let x = 0; x < width; x += 44) {
      context.beginPath()
      context.moveTo(x, 0)
      context.lineTo(x, height)
      context.stroke()
    }

    for (let y = 0; y < height; y += 44) {
      context.beginPath()
      context.moveTo(0, y)
      context.lineTo(width, y)
      context.stroke()
    }

    for (let layerIndex = 0; layerIndex < LAYERS.length - 1; layerIndex += 1) {
      for (let fromIndex = 0; fromIndex < LAYERS[layerIndex]; fromIndex += 1) {
        for (let toIndex = 0; toIndex < LAYERS[layerIndex + 1]; toIndex += 1) {
          const weight = this.engine.weights[layerIndex][fromIndex][toIndex]
          const from = positions[layerIndex][fromIndex]
          const to = positions[layerIndex + 1][toIndex]
          const absoluteWeight = Math.abs(weight)
          const color = getWeightColorCss(weight)
          const seed = layerIndex * 100 + fromIndex * 10 + toIndex
          const controlX = (from.x + to.x) / 2
          const controlY = (from.y + to.y) / 2 + Math.sin(seed * 0.513) * 16

          context.beginPath()
          context.moveTo(from.x, from.y)
          context.quadraticCurveTo(controlX, controlY, to.x, to.y)
          context.strokeStyle = color
          context.lineWidth = 0.8 + absoluteWeight * 5
          context.globalAlpha = 0.12 + absoluteWeight * 0.22
          context.stroke()

          context.beginPath()
          context.moveTo(from.x, from.y)
          context.quadraticCurveTo(controlX, controlY, to.x, to.y)
          context.lineWidth = 0.4 + absoluteWeight * 1.8
          context.globalAlpha = 0.55 + absoluteWeight * 0.3
          context.stroke()

          const dashOffset = (this.phase2d * 0.6 + seed * 0.3) % 1

          for (let dashIndex = 0; dashIndex < DASH_COUNT; dashIndex += 1) {
            const dashT = (dashOffset + dashIndex * DASH_GAP) % 1
            const dashEnd = Math.min(1, dashT + 0.06)
            const x0 = this.quadraticAt(from.x, controlX, to.x, dashT)
            const y0 = this.quadraticAt(from.y, controlY, to.y, dashT)
            const x1 = this.quadraticAt(from.x, controlX, to.x, dashEnd)
            const y1 = this.quadraticAt(from.y, controlY, to.y, dashEnd)
            const fade = Math.sin(dashT * Math.PI)

            context.beginPath()
            context.moveTo(x0, y0)
            context.lineTo(x1, y1)
            context.strokeStyle = color
            context.lineWidth = 1.5 + absoluteWeight * 3.5
            context.globalAlpha = (0.7 + absoluteWeight * 0.28) * fade
            context.shadowColor = color
            context.shadowBlur = 6 + absoluteWeight * 8
            context.stroke()
            context.shadowBlur = 0
          }

          context.globalAlpha = 1
          context.font = 'bold 12px IBM Plex Mono, monospace'
          context.fillStyle = color
          context.textAlign = 'center'
          context.globalAlpha = 0.85
          context.fillText(
            `${weight >= 0 ? '+' : ''}${weight.toFixed(3)}`,
            controlX,
            controlY - 10,
          )
          context.globalAlpha = 1
        }
      }
    }

    for (let layerIndex = 0; layerIndex < LAYERS.length; layerIndex += 1) {
      for (let neuronIndex = 0; neuronIndex < LAYERS[layerIndex]; neuronIndex += 1) {
        const point = positions[layerIndex][neuronIndex]
        const activation = this.engine.activations[layerIndex][neuronIndex]
        const radius = 18 + activation * 5
        const rgb = activation > 0.1 ? '74,144,217' : '208,80,112'
        const glow = context.createRadialGradient(
          point.x,
          point.y,
          radius * 0.3,
          point.x,
          point.y,
          radius * 2.6,
        )

        glow.addColorStop(0, `rgba(${rgb},${0.08 + activation * 0.16})`)
        glow.addColorStop(1, `rgba(${rgb},0)`)
        context.fillStyle = glow
        context.beginPath()
        context.arc(point.x, point.y, radius * 2.6, 0, Math.PI * 2)
        context.fill()

        context.beginPath()
        context.arc(point.x, point.y, radius, 0, Math.PI * 2)
        const gradient = context.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius)
        gradient.addColorStop(0, `hsl(210,50%,${20 + activation * 28}%)`)
        gradient.addColorStop(1, `hsl(210,50%,${8 + activation * 10}%)`)
        context.fillStyle = gradient
        context.fill()
        context.strokeStyle = `rgba(${rgb},${0.5 + activation * 0.45})`
        context.lineWidth = 1.5
        context.stroke()
        context.fillStyle = `rgba(200,220,240,${0.75 + activation * 0.25})`
        context.font = 'bold 11px IBM Plex Mono, monospace'
        context.textAlign = 'center'
        context.textBaseline = 'middle'
        context.fillText(activation.toFixed(2), point.x, point.y)
        context.textBaseline = 'alphabetic'
      }
    }

    const labelY = this.canvas2d.height * 0.18

    for (let layerIndex = 0; layerIndex < LAYERS.length; layerIndex += 1) {
      const x = positions[layerIndex][0].x

      context.font = '500 13px IBM Plex Mono, monospace'
      context.fillStyle = '#6ab0f0'
      context.textAlign = 'center'
      context.shadowColor = '#3a80d0'
      context.shadowBlur = 6
      context.fillText(LAYER_NAMES[layerIndex].toUpperCase(), x, labelY - 20)
      context.font = '11px IBM Plex Mono, monospace'
      context.fillStyle = '#4a6878'
      context.shadowBlur = 0
      context.fillText(LAYER_FUNCTIONS[layerIndex], x, labelY - 5)
    }

    context.font = '400 13px IBM Plex Mono, monospace'
    context.fillStyle = '#5a9fd0'
    context.textAlign = 'left'
    context.fillText('NEURAL NETWORK — 2D', 20, 26)
    context.font = '11px IBM Plex Mono, monospace'
    context.fillStyle = '#2a4a60'
    context.fillText(ARCHITECTURE_LABEL, 20, 44)
  }

  disposeScene() {
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

    if (this.renderer?.domElement?.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement)
    }

    this.scene = null
    this.camera = null
    this.renderer = null
    this.networkGroup = null
    this.neurons = []
    this.connections = []
    this.weightSprites = []
    this.layerSprites = []
    this.selectedNeuron = null
  }
}
