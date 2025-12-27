import * as d3 from 'd3';

/**
 * 在一个容器内同时生成柱状图、扇形图和折线图
 * @param {Array<Array<string|number>>} data - 选中区域的二维数据
 * @param {HTMLElement|string} container - 容器元素或选择器
 */
export function vis(data, container = '#visualization-container') {
  if (!data || data.length === 0) {
    console.warn('visualizeData: 没有数据可视化');
    return;
  }

  const el = typeof container === 'string' ? document.querySelector(container) : container;
  if (!el) return;

  // --- 1. 数据处理 ---
  const allValues = data.flat();
  const frequencyMap = {};
  allValues.forEach(value => {
    const key = String(value);
    frequencyMap[key] = (frequencyMap[key] || 0) + 1;
  });

  // 转换为易于 D3 处理的对象数组
  const chartData = Object.entries(frequencyMap).map(([label, value]) => ({ label, value }));
  if (chartData.length === 0) {
    el.innerHTML = '<p>没有可可视化的数据</p>';
    return;
  }

  // --- 2. 容器准备 (设置为多列布局) ---
  el.innerHTML = '';
  el.style.display = 'flex';
  el.style.flexWrap = 'wrap';
  el.style.gap = '20px';
  el.style.justifyContent = 'space-around';

  const width = 350;
  const height = 300;
  const margin = { top: 30, right: 20, bottom: 60, left: 50 };

  // 创建三个子容器
  const types = ['bar', 'pie', 'line'];
  const subContainers = types.map(type => {
    const div = document.createElement('div');
    div.className = `chart-item chart-${type}`;
    el.appendChild(div);
    return d3.select(div);
  });

  // --- 3. 绘制函数 ---

  // A. 绘制柱状图
  drawBar(subContainers[0], chartData, width, height, margin);

  // B. 绘制扇形图 (饼图)
  drawPie(subContainers[1], chartData, width, height);

  // C. 绘制折线图
  drawLine(subContainers[2], chartData, width, height, margin);
}

/** 柱状图逻辑 **/
function drawBar(selection, data, width, height, margin) {
  const svg = selection.append('svg').attr('width', width).attr('height', height);
  
  const x = d3.scaleBand()
    .domain(data.map(d => d.label))
    .range([margin.left, width - margin.right])
    .padding(0.2);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)]).nice()
    .range([height - margin.bottom, margin.top]);

  svg.append('g')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x))
    .selectAll("text").attr("transform", "rotate(-45)").style("text-anchor", "end");

  svg.append('g')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y));

  svg.selectAll('rect')
    .data(data)
    .enter().append('rect')
    .attr('x', d => x(d.label))
    .attr('y', d => y(d.value))
    .attr('width', x.bandwidth())
    .attr('height', d => height - margin.bottom - y(d.value))
    .attr('fill', '#69b3a2');
    
  svg.append('text').attr('x', width/2).attr('y', 20).attr('text-anchor', 'middle').text('频次柱状图').style('font-weight', 'bold');
}

/** 扇形图逻辑 **/
function drawPie(selection, data, width, height) {
  const svg = selection.append('svg').attr('width', width).attr('height', height);
  const radius = Math.min(width, height) / 2 - 40;
  const color = d3.scaleOrdinal(d3.schemeTableau10);

  const g = svg.append('g').attr('transform', `translate(${width / 2},${height / 2 + 10})`);

  const pie = d3.pie().value(d => d.value);
  const arc = d3.arc().innerRadius(0).outerRadius(radius);

  const arcs = g.selectAll('.arc')
    .data(pie(data))
    .enter().append('g');

  arcs.append('path')
    .attr('d', arc)
    .attr('fill', (d, i) => color(i))
    .attr('stroke', '#fff');

  // 简单的标签显示（仅显示值较大的）
  arcs.filter(d => d.endAngle - d.startAngle > 0.2)
    .append('text')
    .attr('transform', d => `translate(${arc.centroid(d)})`)
    .attr('text-anchor', 'middle')
    .text(d => d.data.label)
    .style('font-size', '10px').style('fill', '#fff');

  svg.append('text').attr('x', width/2).attr('y', 20).attr('text-anchor', 'middle').text('占比扇形图').style('font-weight', 'bold');
}

/** 折线图逻辑 **/
function drawLine(selection, data, width, height, margin) {
  const svg = selection.append('svg').attr('width', width).attr('height', height);

  const x = d3.scalePoint()
    .domain(data.map(d => d.label))
    .range([margin.left, width - margin.right]);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)]).nice()
    .range([height - margin.bottom, margin.top]);

  svg.append('g').attr('transform', `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x));
  svg.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y));

  const line = d3.line()
    .x(d => x(d.label))
    .y(d => y(d.value))
    .curve(d3.curveMonotoneX); // 使线条圆滑

  svg.append('path')
    .datum(data)
    .attr('fill', 'none')
    .attr('stroke', '#4e79a7')
    .attr('stroke-width', 2)
    .attr('d', line);

  svg.selectAll('circle')
    .data(data)
    .enter().append('circle')
    .attr('cx', d => x(d.label))
    .attr('cy', d => y(d.value))
    .attr('r', 4)
    .attr('fill', '#4e79a7');

  svg.append('text').attr('x', width/2).attr('y', 20).attr('text-anchor', 'middle').text('趋势折线图').style('font-weight', 'bold');
}