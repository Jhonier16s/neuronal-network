import { Fragment } from 'react'

const LAYERS = [2, 3, 2, 1]

function EntryOverlay({ onEnter }) {
  return (
    <div id="entry">
      <h2>AI Training Chamber</h2>
      <div className="entry-kicker">AI Training Chamber</div>

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
        Room tour cinematografico y exploracion libre
        <br />
        Racks vivos · banco de hardware · prototipos roboticos · nucleo XOR
      </p>

      <button id="ebtn" onClick={onEnter}>
        Iniciar room tour
      </button>
    </div>
  )
}

export default EntryOverlay
