import { Fragment, useMemo, useState } from 'react'

function parseLayerList(value) {
  const parts = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

  if (parts.length < 2) {
    return { error: 'Define al menos entrada y salida.' }
  }

  const layers = parts.map((item) => Number(item))

  if (layers.some((item) => !Number.isInteger(item) || item < 1 || item > 8)) {
    return { error: 'Cada capa debe ser un entero entre 1 y 8.' }
  }

  if (layers.length > 6) {
    return { error: 'Usa maximo 6 capas para que la room siga legible.' }
  }

  return { value: layers }
}

function parseVector(value) {
  const parts = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

  if (parts.length === 0) {
    return { error: 'Completa este vector.' }
  }

  const numbers = parts.map((item) => Number(item))

  if (numbers.some((item) => Number.isNaN(item))) {
    return { error: 'Usa solo numeros separados por comas.' }
  }

  return { value: numbers }
}

function formatVector(values) {
  return values.join(', ')
}

function EntryOverlay({ initialConfig, onEnter }) {
  const initialSample = initialConfig.trainingData[0]
  const [architectureText, setArchitectureText] = useState(initialConfig.layers.join(', '))
  const [inputText, setInputText] = useState(formatVector(initialSample.input))
  const [targetText, setTargetText] = useState(formatVector(initialSample.target))

  const formState = useMemo(() => {
    const layersResult = parseLayerList(architectureText)
    const inputResult = parseVector(inputText)
    const targetResult = parseVector(targetText)
    const layers = layersResult.value ?? initialConfig.layers
    let error = layersResult.error ?? inputResult.error ?? targetResult.error ?? ''

    if (!error && inputResult.value.length !== layers[0]) {
      error = `La entrada debe tener ${layers[0]} valores.`
    }

    if (!error && targetResult.value.length !== layers[layers.length - 1]) {
      error = `La salida debe tener ${layers[layers.length - 1]} valores.`
    }

    return {
      layers,
      input: inputResult.value ?? initialSample.input,
      target: targetResult.value ?? initialSample.target,
      error,
      valid: !error,
    }
  }, [architectureText, initialConfig.layers, initialSample.input, initialSample.target, inputText, targetText])

  return (
    <div id="entry">
      <h2>AI Training Chamber</h2>
      <div className="entry-kicker">Configura tu modelo antes de entrar</div>

      <div className="arch" aria-hidden="true">
        {formState.layers.map((count, index) => (
          <Fragment key={`${count}-${index}`}>
            <div className="alay">
              {Array.from({ length: count }, (_, itemIndex) => (
                <div className="adot" key={itemIndex} />
              ))}
            </div>
            {index < formState.layers.length - 1 ? <div className="aln" /> : null}
          </Fragment>
        ))}
      </div>

      <p>
        Define la arquitectura, el vector de entrada y el vector objetivo.
        <br />
        El entrenamiento visual se mantiene, pero ahora la red nace desde tu configuracion.
      </p>

      <div className="entry-form">
        <label className="entry-field">
          <span>Arquitectura</span>
          <input
            value={architectureText}
            onChange={(event) => setArchitectureText(event.target.value)}
            placeholder="2, 3, 2, 1"
          />
        </label>

        <label className="entry-field">
          <span>Vector de entrada</span>
          <input
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            placeholder="0.7, 0.6"
          />
        </label>

        <label className="entry-field">
          <span>Vector objetivo</span>
          <input
            value={targetText}
            onChange={(event) => setTargetText(event.target.value)}
            placeholder="0.1"
          />
        </label>

        <div className={`entry-status ${formState.valid ? 'ok' : 'error'}`}>
          {formState.valid
            ? `Listo: ${formState.layers.join(' · ')} | entrada ${formState.input.length} | salida ${formState.target.length}`
            : formState.error}
        </div>
      </div>

      <button
        id="ebtn"
        onClick={() => onEnter(formState)}
        disabled={!formState.valid}
      >
        Iniciar room tour
      </button>
    </div>
  )
}

export default EntryOverlay
