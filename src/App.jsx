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

      <PointerLockPrompt visible={!viewState.pointerLocked && !viewState.cinematicActive} />
      <ControlsHelp />

      <TrainingPanel
        training={viewState.training}
        cinematicActive={viewState.cinematicActive}
        onStepBack={() => controller?.stepBack()}
        onReset={() => controller?.resetTraining()}
        onStepForward={() => controller?.stepForward()}
        onStepEpochForward={() => controller?.stepEpochForward()}
      />

      {viewState.cinematicActive ? null : (
        <div id="arrow-hint">
          <b>&larr; &rarr;</b> navegar pasos · <b>ESPACIO</b> auto-entrenar
        </div>
      )}

      {viewState.entryVisible ? (
        <EntryOverlay onEnter={() => controller?.enterScene()} />
      ) : null}
    </div>
  )
}

export default App
