//用于从选中的单元格中获得数据
import Sheet from "../component/sheet";

// src/visualization/get_data.js

export function get_data(sheet) {
  try {
    const { data, selector } = sheet;

    if (!data || !selector) {
      console.error('getSelectedData: sheet 未定义或缺少必要属性');
      return [];
    }

    // 获取选区范围
    const range = selector.range;
    if (!range) {
      console.error('getSelectedData: 未找到选区');
      return [];
    }

    const { sri, sci, eri, eci } = range; // 起始行/列，结束行/列

    const result = [];
    for (let r = sri; r <= eri; r++) {
      const row = [];
      for (let c = sci; c <= eci; c++) {
        const cell = data.getCell(r, c);
        if (cell) {
          row.push(cell.text || cell.value || '');
        } else {
          row.push('');
        }
      }
      result.push(row);
    }

    console.log('选中区域数据：', result);
    return result;
  } catch (err) {
    console.error('get_data.js 出错：', err);
    return [];
  }
}
