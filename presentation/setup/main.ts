import { defineAppSetup } from '@slidev/types'
// @ts-expect-error deep import into Slidev internals
import { showGotoDialog, showOverview } from '@slidev/client/state/index.ts'

// Force the overview/menu overlay to always close on Escape or `o`,
// regardless of focus or drawing-mode state. The stock Slidev shortcut
// guards both keys with `not(drawingEnabled)`, which sometimes leaves
// the overlay stuck open; a capture-phase document listener bypasses it.
export default defineAppSetup(() => {
  if (typeof document === 'undefined') return

  // Hide the Goto-dialog autocomplete list while the dialog is closed.
  // Slidev v0.49.29 keeps the parent #slidev-goto-dialog in DOM (offset
  // offscreen with class `-top-20`), but its autocomplete-list child
  // remains positioned relative and leaks back into the viewport as a
  // persistent right-side strip of slide titles. Scope the hide by the
  // offscreen class so the list only appears when the user opens goto.
  const style = document.createElement('style')
  style.textContent = `
    #slidev-goto-dialog.-top-20 .autocomplete-list { display: none !important; }
  `
  document.head.appendChild(style)

  const isEditable = (el: EventTarget | null): boolean => {
    if (!(el instanceof HTMLElement)) return false
    const tag = el.tagName
    return (
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      el.isContentEditable
    )
  }

  document.addEventListener(
    'keydown',
    (e) => {
      if (isEditable(e.target)) return
      if (e.key === 'Escape') {
        if (showOverview.value || showGotoDialog.value) {
          showOverview.value = false
          showGotoDialog.value = false
          e.preventDefault()
          e.stopPropagation()
        }
        return
      }
      if (
        (e.key === 'o' || e.key === 'O' || e.key === '`') &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey
      ) {
        showOverview.value = !showOverview.value
        e.preventDefault()
        e.stopPropagation()
      }
    },
    true,
  )
})
