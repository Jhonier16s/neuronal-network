function PointerLockPrompt({ visible }) {
  return (
    <div id="lock-prompt" className={visible ? '' : 'hidden'}>
      <span className="lock-prompt-accent">Haz clic para entrar y moverte</span>
      {' · '}<b>WASD</b>{' '}para caminar{' · '}<b>ESC</b>{' '}para soltar
    </div>
  )
}

export default PointerLockPrompt
