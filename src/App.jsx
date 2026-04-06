import { useEffect, useRef, useState } from 'react'
import './App.css'
import ControlsHelp from './components/ControlsHelp.jsx'
import EntryOverlay from './components/EntryOverlay.jsx'
import NeuronInspector from './components/NeuronInspector.jsx'
import PointerLockPrompt from './components/PointerLockPrompt.jsx'
import TopBar from './components/TopBar.jsx'
import TrainingPanel from './components/TrainingPanel.jsx'
import {
  NeuralRoomController,
  createInitialViewState,
} from './neural-room/controller.js'

function App() {
  const viewportRef = useRef(null)
  const canvas2dRef = useRef(null)
  const controllerRef = useRef(null)
  const [viewState, setViewState] = useState(() => createInitialViewState())

  useEffect(() => {
    const controller = new NeuralRoomController({
      onStateChange: (nextState) => {
        setViewState(nextState)
      },
    })

    controllerRef.current = controller
    controller.mount({
      viewportEl: viewportRef.current,
      canvas2d: canvas2dRef.current,
    })

    return () => {
      controller.unmount()
      controllerRef.current = null
    }
  }, [])

  const controller = controllerRef.current

  return (
    <div className="app-shell">
      <div id="scene-root" ref={viewportRef} aria-hidden="true" />

      <TopBar
        architectureLabel={viewState.architectureLabel}
        connectionCount={viewState.connectionCount}
        epoch={viewState.epoch}
        mouseLabel={viewState.mouseLabel}
      />

      <NeuronInspector
        neuron={viewState.selectedNeuron}
        visible={viewState.infoVisible}
      />

      <div id="xhair" className={viewState.pointerLocked ? 'on' : ''} />

      <PointerLockPrompt visible={!viewState.pointerLocked} />
      <ControlsHelp />

      {!viewState.entryVisible && viewState.mode3D ? (
        <button id="mode-btn" onClick={() => controller?.toggleMode()}>
          Vista 2D
        </button>
      ) : null}

      <TrainingPanel
        training={viewState.training}
        onStepBack={() => controller?.stepBack()}
        onStepForward={() => controller?.stepForward()}
      />

      <div id="arrow-hint">
        <b>&larr; &rarr;</b> navegar pasos · <b>ESPACIO</b> auto-entrenar
      </div>

      <div id="c2wrap" className={!viewState.mode3D ? 'show' : ''}>
        <canvas id="c2d" ref={canvas2dRef} />
        {!viewState.mode3D ? (
          <button id="back-btn" onClick={() => controller?.toggleMode()}>
            &larr; Vista 3D
          </button>
        ) : null}
      </div>

      {viewState.entryVisible ? (
        <EntryOverlay onEnter={() => controller?.enterScene()} />
      ) : null}
    </div>
  )
}

export default App
