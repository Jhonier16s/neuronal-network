import LossChart from './LossChart.jsx'

function TrainingPanel({ training, onStepBack, onStepForward }) {
  return (
    <div id="train-panel">
      <h3>Entrenamiento XOR</h3>

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

      <LossChart
        losses={training.losses}
        currentIndex={training.currentLossIndex}
      />

      <div id="xor-outputs">
        {training.outputs.map((output) => (
          <div className="xor-row" key={output.label}>
            <span className="xor-in">{output.label}</span>
            <span
              className={`xor-out ${
                output.correct === null ? '' : output.correct ? 'correct' : 'wrong'
              }`}
            >
              {output.value}
            </span>
          </div>
        ))}
      </div>

      <div id="step-controls">
        <button className="step-btn" onClick={onStepBack}>
          &larr; Atras
        </button>
        <span id="step-pos">{training.stepPosition}</span>
        <button className="step-btn primary" onClick={onStepForward}>
          &rarr; Siguiente
        </button>
      </div>
    </div>
  )
}

export default TrainingPanel
