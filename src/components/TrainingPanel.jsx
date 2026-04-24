import LossChart from './LossChart.jsx'

function TrainingPanel({
  training,
  onBackward,
  onForward,
  onToggleAuto,
  onAutoSpeedChange,
  onReset,
  cinematicActive,
}) {
  return (
    <div id="train-panel">
      <h3>Entrenamiento activo</h3>

      <div className="train-row">
        <span className="train-lbl">Paso</span>
        <span className="train-val">{training.step}</span>
      </div>
      <div className="train-row">
        <span className="train-lbl">Loss</span>
        <span className={`train-val ${training.lossTone}`}>{training.lossText}</span>
      </div>
      <div className="train-row">
        <span className="train-lbl">LR</span>
        <span className="train-val">{training.lrText}</span>
      </div>
      <div className="train-row">
        <span className="train-lbl">Estado</span>
        <span className={`train-val ${training.statusTone}`}>{training.statusText}</span>
      </div>
      <div className="train-row train-row-stack">
        <span className="train-lbl">Muestra</span>
        <span className="train-val mono">{training.sampleText}</span>
      </div>
      <div className="train-row train-row-stack">
        <span className="train-lbl">Fase</span>
        <span className="train-val mono">{training.phaseText}</span>
      </div>
      <div className="train-row train-row-stack">
        <span className="train-lbl">Visual</span>
        <span className="train-val mono dimmed">{training.visualText}</span>
      </div>

      <LossChart losses={training.losses} currentIndex={training.currentLossIndex} />

      <div className="train-row train-row-stack train-speed-row">
        <div className="train-speed-head">
          <span className="train-lbl">Velocidad auto</span>
          <span className="train-val mono">{training.autoSpeed.toFixed(2)} pasos/s</span>
        </div>
        <input
          className="train-speed-slider"
          type="range"
          min="0.25"
          max="15"
          step="0.25"
          value={training.autoSpeed}
          onChange={(event) => onAutoSpeedChange(event.target.value)}
          disabled={cinematicActive}
        />
      </div>

      <div id="step-controls">
        <button
          className="step-btn step-btn-backward"
          onClick={onBackward}
          disabled={cinematicActive || !training.canBackward}
        >
          Backward
        </button>
        <button className="step-btn step-btn-reset" onClick={onReset} disabled={cinematicActive}>
          Reiniciar
        </button>
        <button
          className="step-btn primary"
          onClick={onForward}
          disabled={cinematicActive || !training.canForward}
        >
          Forward
        </button>
      </div>

      <button
        className="step-btn step-btn-epoch"
        onClick={onToggleAuto}
        disabled={cinematicActive}
      >
        {training.autoActive ? 'Detener auto' : 'Automatico'}
      </button>

      <div id="step-pos">{training.stepPosition}</div>
    </div>
  )
}

export default TrainingPanel
