import { useEffect, useRef, useState } from 'react'
import './App.css'
import ControlsHelp from './components/ControlsHelp.jsx'
import EntryOverlay from './components/EntryOverlay.jsx'
import NeuronInspector from './components/NeuronInspector.jsx'
import PointerLockPrompt from './components/PointerLockPrompt.jsx'
import TopBar from './components/TopBar.jsx'
import TrainingPanel from './components/TrainingPanel.jsx'
import { buildModelConfig, createDefaultModelConfig } from './neural-room/model-config.js'
import { createInitialViewState } from './neural-room/view-state.js'

function App() {
  const viewportRef = useRef(null)
  const controllerRef = useRef(null)
  const queuedEnterConfigRef = useRef(null)
  const [initialModelConfig] = useState(() => createDefaultModelConfig())
  const [viewState, setViewState] = useState(() => createInitialViewState(initialModelConfig))
  const [sceneReady, setSceneReady] = useState(false)

  useEffect(() => {
    let cancelled = false

    setSceneReady(false)

    import('./neural-room/controller.js')
      .then(({ NeuralRoomController }) => {
        if (cancelled) {
          return null
        }

        const controller = new NeuralRoomController({
          modelConfig: initialModelConfig,
          onStateChange: (nextState) => {
            setViewState(nextState)
          },
        })

        controllerRef.current = controller

        return controller.mount({
          viewportEl: viewportRef.current,
        }).then(() => controller)
      })
      .then((controller) => {
        if (controller && queuedEnterConfigRef.current) {
          controller.enterScene(queuedEnterConfigRef.current)
          queuedEnterConfigRef.current = null
        }
      })
      .then(() => {
        if (!cancelled) {
          setSceneReady(true)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSceneReady(true)
        }
      })

    return () => {
      cancelled = true
      controllerRef.current?.unmount()
      controllerRef.current = null
    }
  }, [initialModelConfig])

  const controller = controllerRef.current
  const handleEnter = ({ layers, input, target }) => {
    const modelConfig = buildModelConfig({ layers, input, target })

    if (controller) {
      controller.enterScene(modelConfig)
      return
    }

    queuedEnterConfigRef.current = modelConfig
  }

  return (
    <div className="app-shell">
      <div id="scene-root" ref={viewportRef} aria-hidden="true" />
      <div className={`scene-loader ${sceneReady ? 'hidden' : ''}`}>
        <div className="scene-loader-chip">Optimizando escena</div>
        <h2>Neural Room</h2>
        <p>Preparando shaders, buffers y postproceso para que la entrada sea mas fluida.</p>
      </div>

      <TopBar
        architectureLabel={viewState.architectureLabel}
        connectionCount={viewState.connectionCount}
        epoch={viewState.epoch}
        hintActive={!viewState.pointerLocked && !viewState.cinematicActive}
        mouseLabel={viewState.mouseLabel}
      />

      <NeuronInspector
        neuron={viewState.selectedNeuron}
        visible={viewState.infoVisible}
      />

      <div id="xhair" className={viewState.pointerLocked ? 'on' : ''} />

      <PointerLockPrompt visible={!viewState.pointerLocked && !viewState.cinematicActive} />
      <ControlsHelp />

      <TrainingPanel
        training={viewState.training}
        cinematicActive={viewState.cinematicActive}
        onBackward={() => controller?.stepBackward()}
        onAutoSpeedChange={(value) => controller?.setAutoTrainSpeed(value)}
        onReset={() => controller?.resetTraining()}
        onForward={() => controller?.stepForward()}
        onToggleAuto={() => controller?.toggleAutoTrain()}
      />

      {viewState.cinematicActive ? null : (
        <div id="arrow-hint">
          <b>&rarr;</b> forward · <b>&larr;</b> backward · <b>ESPACIO</b> auto
        </div>
      )}

      {viewState.entryVisible ? (
        <EntryOverlay
          initialConfig={initialModelConfig}
          onEnter={handleEnter}
        />
      ) : null}
    </div>
  )
}

export default App
