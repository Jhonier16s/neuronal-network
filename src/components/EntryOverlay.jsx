import { Fragment } from 'react'

const LAYERS = [2, 3, 2, 1]

function EntryOverlay({ onEnter }) {
  return (
    <div id="entry">
      <h2>Neural Network</h2>

      <div className="arch" aria-hidden="true">
        {LAYERS.map((count, index) => (
          <Fragment key={`${count}-${index}`}>
            <div className="alay">
              {Array.from({ length: count }, (_, itemIndex) => (
                <div className="adot" key={itemIndex} />
              ))}
            </div>
            {index < LAYERS.length - 1 ? <div className="aln" /> : null}
          </Fragment>
        ))}
      </div>

      <p>
        Arquitectura 2 · 3 · 2 · 1
        <br />
        Arrastra para rotar · Scroll para zoom · Clic en neurona
      </p>

      <button id="ebtn" onClick={onEnter}>
        Entrar
      </button>
    </div>
  )
}

export default EntryOverlay
