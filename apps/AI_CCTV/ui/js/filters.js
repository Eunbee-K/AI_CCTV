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
//
// 신뢰도는 **행 단위**로 rowVisible이 거른다. 여기서 또 걸면 같은 기준이
// 두 군데서 적용돼, 행은 보이는데 박스만 사라지는 어정쩡한 상태가 된다.
// 여기서는 클래스 토글만 본다.
export function filterBoxes(boxes) {
  return (boxes || []).filter((b) => isClassEnabled(b.label));
}

// 행 표시 여부.
//
// **표의 '신뢰도' 열과 같은 값으로 거른다.** 예전에는 YOLO 박스만 봐서,
// 분류기가 이름을 붙인 행(박스가 없다)은 슬라이더를 올려도 사라지지 않았다 —
// 화면에 보이는 신뢰도와 필터가 따로 놀았다.
//
// 신뢰도가 없는 행(이름 미부여·수동 추가)은 슬라이더가 0일 때만 보인다.
// 슬라이더를 올린 것은 "확신 있는 것만 보겠다"는 뜻이라, 값이 없는 행을
// 계속 남기면 걸러지지 않는 것처럼 보인다.
export function rowVisible(row) {
  // 사람이 직접 넣은 행은 신뢰도가 없는 게 정상이다. 검수자가 손으로 넣은 것을
  // 슬라이더가 숨겨버리면 작업물을 잃는다.
  if (!row.manual) {
    const conf = Number(row.conf);
    if (Number.isFinite(conf) && conf > 0) {
      if (conf < state.confMin) return false;
    } else if (state.confMin > 0) {
      return false;
    }
  }
  const boxes = row.boxes || [];
  if (!boxes.length) return true;          // 박스가 없어도 클래스 필터는 못 건다
  return filterBoxes(boxes).length > 0;    // 클래스 토글은 박스 기준 그대로
}
