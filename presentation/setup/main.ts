import { defineAppSetup } from '@slidev/types'
// @ts-expect-error deep import into Slidev internals
import { showGotoDialog, showOverview } from '@slidev/client/state/index.ts'

// Force the overview/menu overlay to always close on Escape or `o`,
// regardless of focus or drawing-mode state. The stock Slidev shortcut
// guards both keys with `not(drawingEnabled)`, which sometimes leaves
// the overlay stuck open; a capture-phase document listener bypasses it.
export default defineAppSetup(() => {
  if (typeof document === 'undefined') return

  // Slidev v0.49.29 ships Goto.vue with `v-if="result.length > 0"` on the
  // autocomplete list. Fuse.js v7 returns *every* item for an empty query,
  // so the list is mounted at app start and sits as a persistent strip of
  // slide titles leaking past the offscreen dialog. Hide it by default and
  // only reveal it when the parent actually has the open-state `top-5`
  // class. Also relocate the dialog to the left edge and give it a max
  // width so it never overlaps slide content.
  const style = document.createElement('style')
  style.textContent = `
    #slidev-goto-dialog { right: auto !important; left: 1.25rem !important; max-width: 18rem !important; width: 18rem !important; min-width: 0 !important; }
    #slidev-goto-dialog .autocomplete-list { display: none !important; }
    #slidev-goto-dialog.top-5 .autocomplete-list { display: block !important; }
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
