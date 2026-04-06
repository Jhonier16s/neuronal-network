function PointerLockPrompt({ visible }) {
  return (
    <div id="lock-prompt" className={visible ? '' : 'hidden'}>
      Clic para capturar raton · <b>ESC</b> para soltar
    </div>
  )
}

export default PointerLockPrompt
