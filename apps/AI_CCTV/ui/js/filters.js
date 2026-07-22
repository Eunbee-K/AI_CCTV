// 신뢰도 임계값 + 클래스 필터 공용 상태.
// results-table(행 숨김)과 player(박스 오버레이)가 같은 필터를 공유한다.

const state = {
  confMin: 0, // 0~1
  enabledClasses: null, // null = 전체 허용, Set = 해당 클래스만
  knownClasses: [],
};

const listeners = new Set();

function emit() {
  for (const cb of listeners) cb();
}

export function onFilterChange(cb) {
  listeners.add(cb);
}

export function getConfMin() {
  return state.confMin;
}

export function setConfMin(v) {
  state.confMin = v;
  emit();
}

export function isClassEnabled(label) {
  return state.enabledClasses === null || state.enabledClasses.has(label);
}

export function toggleClass(label) {
  if (state.enabledClasses === null) {
    // 전체 허용 상태에서 하나를 끄면, 나머지만 허용하는 Set으로 전환
    state.enabledClasses = new Set(state.knownClasses.filter((c) => c !== label));
  } else if (state.enabledClasses.has(label)) {
    state.enabledClasses.delete(label);
  } else {
    state.enabledClasses.add(label);
    if (state.enabledClasses.size === state.knownClasses.length) {
      state.enabledClasses = null; // 전부 켜지면 다시 "전체 허용"
    }
  }
  emit();
}

export function setKnownClasses(classes) {
  state.knownClasses = [...classes];
  // 새 클래스가 나타났을 때 기존 Set에 자동 포함되도록 전체 허용이면 그대로 둠
}

export function getKnownClasses() {
  return state.knownClasses;
}

// 박스 목록에 필터 적용 (오버레이용)
export function filterBoxes(boxes) {
  return (boxes || []).filter(
    (b) =>
      (typeof b.confidence !== "number" || b.confidence >= state.confMin) &&
      isClassEnabled(b.label)
  );
}

// 행 표시 여부: 박스가 있으면 필터 통과 박스가 하나라도 있어야 표시,
// 박스가 없는 행(수동 추가 등)은 항상 표시.
export function rowVisible(row) {
  const boxes = row.boxes || [];
  if (!boxes.length) return true;
  return filterBoxes(boxes).length > 0;
}
