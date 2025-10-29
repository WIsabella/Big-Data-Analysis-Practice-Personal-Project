//用于将获得的数据进行可视化
// src/visualization/vis.js
import * as d3 from 'd3';

/**
 * 使用 D3 可视化选中数据
 * @param {Array<Array<string|number>>} data - 选中区域的二维数据
 * @param {HTMLElement|string} container - 容器元素或选择器
 */
export function vis(data, container = '#visualization-container') {
  if (!data || data.length === 0) {
    console.warn('visualizeData: 没有数据可视化');
    return;
  }

  const el = typeof container === 'string' ? document.querySelector(container) : container;
  if (!el) {
    console.error('visualizeData: 无法找到容器元素');
    return;
  }

  // 清空旧图表
  el.innerHTML = '';

  // 扁平化数据并统计频率
  const allValues = data.flat(); // 跳过表头并扁平化
  const frequencyMap = {};
  
  // 统计每个值的出现次数
  allValues.forEach(value => {
    const key = String(value); // 统一转为字符串作为键
    frequencyMap[key] = (frequencyMap[key] || 0) + 1;
  });

  // 准备图表数据
  const xLabels = Object.keys(frequencyMap);
  const yValues = Object.values(frequencyMap);

  // 如果没有数据，直接返回
  if (xLabels.length === 0) {
    el.innerHTML = '<p>没有可可视化的数据</p>';
    return;
  }

  const width = 400;
  const height = 300;
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };

  const svg = d3.select(el)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  // X轴 - 分类数据
  const x = d3.scaleBand()
    .domain(xLabels)
    .range([margin.left, width - margin.right])
    .padding(0.2);

  // Y轴 - 数值数据
  const y = d3.scaleLinear()
    .domain([0, d3.max(yValues)])  // 修正：应该是 yValues 的最大值
    .nice()
    .range([height - margin.bottom, margin.top]);

  // 添加X轴
  svg.append('g')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x))
    .selectAll("text")
    .style("text-anchor", "end")
    .attr("dx", "-.8em")
    .attr("dy", ".15em")
    .attr("transform", "rotate(-45)");

  // 添加Y轴
  svg.append('g')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y));

  // 添加柱状图
  svg.selectAll('rect')
    .data(yValues)
    .enter()
    .append('rect')
    .attr('x', (d, i) => x(xLabels[i]))
    .attr('y', d => y(d))
    .attr('width', x.bandwidth())
    .attr('height', d => height - margin.bottom - y(d))  // 修正高度计算
    .attr('fill', '#69b3a2');

  // 添加数值标签
  svg.selectAll('.value-label')
    .data(yValues)
    .enter()
    .append('text')
    .attr('class', 'value-label')
    .attr('x', (d, i) => x(xLabels[i]) + x.bandwidth() / 2)
    .attr('y', d => y(d) - 5)
    .attr('text-anchor', 'middle')
    .text(d => d)
    .style('font-size', '12px')
    .style('fill', '#333');
}