--Edited s.t. no subqueries occur, i.e., queries with correlated subqueries were removed (queries 2,16,17,20), for uncorrelated ones we replaced the subquery with the result (queries 11,15,22)
--Edit: Queries were rewritten to joins if they contain subqueries (queries 2,16,17,20)

--Query 1
select l_returnflag,
       l_linestatus,
       sum(l_quantity)                                       as sum_qty,
       sum(l_extendedprice)                                  as sum_base_price,
       sum(l_extendedprice * (1 - l_discount))               as sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
       avg(l_quantity)                                       as avg_qty,
       avg(l_extendedprice)                                  as avg_price,
       avg(l_discount)                                       as avg_disc,
       count(*)                                              as count_order
from lineitem
where l_shipdate <= date '1998-12-01' - interval '67' day
group by l_returnflag,
         l_linestatus
order by l_returnflag,
         l_linestatus;

--Query 2
select s_acctbal,
       s_name,
       n_name,
       p_partkey,
       p_mfgr,
       s_address,
       s_phone,
       s_comment
from part,
     supplier,
     partsupp,
     nation,
     region,
     (select ps_partkey, min(ps_supplycost) as ps_supplycost
    from partsupp,
         supplier,
         nation,
         region
    where s_suppkey = ps_suppkey
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_name = 'AFRICA'
    group by ps_partkey) s
where p_partkey = partsupp.ps_partkey
  and s_suppkey = partsupp.ps_suppkey
  and p_size = 16
  and p_type like '%NICKEL'
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'AFRICA'
  and partsupp.ps_supplycost = s.ps_supplycost
  and p_partkey = s.ps_partkey
order by s_acctbal desc,
         n_name,
         s_name,
         p_partkey
limit 100;

--Query 3
select l_orderkey,
       sum(l_extendedprice * (1 - l_discount)) as revenue,
       o_orderdate,
       o_shippriority
from customer,
     orders,
     lineitem
where c_mktsegment = 'BUILDING'
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < date '1995-03-03'
  and l_shipdate > date '1995-03-03'
group by l_orderkey,
         o_orderdate,
         o_shippriority
order by revenue desc,
         o_orderdate
limit 10;

--Query 4
select o_orderpriority,
       count(*) as order_count
from orders
where o_orderdate >= date '1993-07-01'
  and o_orderdate < date '1993-07-01' + interval '3' month
  and exists(
        select *
        from lineitem
        where l_orderkey = o_orderkey
          and l_commitdate < l_receiptdate
    )
group by o_orderpriority
order by o_orderpriority;

--Query 5
select n_name,
       sum(l_extendedprice * (1 - l_discount)) as revenue
from customer,
     orders,
     lineitem,
     supplier,
     nation,
     region
where c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'AFRICA'
  and o_orderdate >= date '1993-01-01'
  and o_orderdate < date '1993-01-01' + interval '1' year
group by n_name
order by revenue desc;

--Query 6
select sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '1993-01-01'
  and l_shipdate < date '1993-01-01' + interval '1' year
  and l_discount between 0.02 - 0.01 and 0.02 + 0.01
  and l_quantity < 24;

--Query 7
select supp_nation,
       cust_nation,
       l_year,
       sum(volume) as revenue
from (
         select n1.n_name                          as supp_nation,
                n2.n_name                          as cust_nation,
                extract(year from l_shipdate)      as l_year,
                l_extendedprice * (1 - l_discount) as volume
         from supplier,
              lineitem,
              orders,
              customer,
              nation n1,
              nation n2
         where s_suppkey = l_suppkey
           and o_orderkey = l_orderkey
           and c_custkey = o_custkey
           and s_nationkey = n1.n_nationkey
           and c_nationkey = n2.n_nationkey
           and (
                 (n1.n_name = 'ALGERIA' and n2.n_name = 'INDIA')
                 or (n1.n_name = 'INDIA' and n2.n_name = 'ALGERIA')
             )
           and l_shipdate between date '1995-01-01' and date '1996-12-31'
     ) as shipping
group by supp_nation,
         cust_nation,
         l_year
order by supp_nation,
         cust_nation,
         l_year;

--Query 8
select o_year,
       sum(case
               when nation = 'INDIA' then volume
               else 0
           end) / sum(volume) as mkt_share
from (
         select extract(year from o_orderdate)     as o_year,
                l_extendedprice * (1 - l_discount) as volume,
                n2.n_name                          as nation
         from part,
              supplier,
              lineitem,
              orders,
              customer,
              nation n1,
              nation n2,
              region
         where p_partkey = l_partkey
           and s_suppkey = l_suppkey
           and l_orderkey = o_orderkey
           and o_custkey = c_custkey
           and c_nationkey = n1.n_nationkey
           and n1.n_regionkey = r_regionkey
           and r_name = 'ASIA'
           and s_nationkey = n2.n_nationkey
           and o_orderdate between date '1995-01-01' and date '1996-12-31'
           and p_type = 'STANDARD POLISHED COPPER'
     ) as all_nations
group by o_year
order by o_year;

--Query 9
select nation,
       o_year,
       sum(amount) as sum_profit
from (
         select n_name                                                          as nation,
                extract(year from o_orderdate)                                  as o_year,
                l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
         from part,
              supplier,
              lineitem,
              partsupp,
              orders,
              nation
         where s_suppkey = l_suppkey
           and ps_suppkey = l_suppkey
           and ps_partkey = l_partkey
           and p_partkey = l_partkey
           and o_orderkey = l_orderkey
           and s_nationkey = n_nationkey
           and p_name like '%burnished%'
     ) as profit
group by nation,
         o_year
order by nation,
         o_year desc;

--Query 10
select c_custkey,
       c_name,
       sum(l_extendedprice * (1 - l_discount)) as revenue,
       c_acctbal,
       n_name,
       c_address,
       c_phone,
       c_comment
from customer,
     orders,
     lineitem,
     nation
where c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate >= date '1993-04-01'
  and o_orderdate < date '1993-04-01' + interval '3' month
  and l_returnflag = 'R'
  and c_nationkey = n_nationkey
group by c_custkey,
         c_name,
         c_acctbal,
         c_phone,
         n_name,
         c_address,
         c_comment
order by revenue desc
limit 20;

--Query 11
select ps_partkey,
       sum(ps_supplycost * ps_availqty) as value
from
    partsupp,
    supplier,
    nation
where
    ps_suppkey = s_suppkey
  and s_nationkey = n_nationkey
  and n_name = 'KENYA'
group by
    ps_partkey
having
    sum(ps_supplycost * ps_availqty) > 7560158.077861000000
order by
    value desc;

--Query 12
select l_shipmode,
       sum(case
               when o_orderpriority = '1-URGENT'
                   or o_orderpriority = '2-HIGH'
                   then 1
               else 0
           end) as high_line_count,
       sum(case
               when o_orderpriority <> '1-URGENT'
                   and o_orderpriority <> '2-HIGH'
                   then 1
               else 0
           end) as low_line_count
from orders,
     lineitem
where o_orderkey = l_orderkey
  and l_shipmode in ('FOB', 'TRUCK')
  and l_commitdate < l_receiptdate
  and l_shipdate < l_commitdate
  and l_receiptdate >= date '1997-01-01'
  and l_receiptdate < date '1997-01-01' + interval '1' year
group by l_shipmode
order by l_shipmode;

--Query 13
select c_count,
       count(*) as custdist
from (
         select c_custkey,
                count(o_orderkey)
         from customer
                  left outer join orders on
                     c_custkey = o_custkey
                 and o_comment not like '%unusual%deposits%'
         group by c_custkey
     ) as c_orders (c_custkey, c_count)
group by c_count
order by custdist desc,
         c_count desc;

--Query 14
select 100.00 * sum(case
                        when p_type like 'PROMO%'
                            then l_extendedprice * (1 - l_discount)
                        else 0
    end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from lineitem,
     part
where l_partkey = p_partkey
  and l_shipdate >= date '1997-10-01'
  and l_shipdate < date '1997-10-01' + interval '1' month;

--Query 15
select s_suppkey,
       s_name,
       s_address,
       s_phone,
       total_revenue
from supplier,
     (select l_suppkey                               AS supplier_no,
             sum(l_extendedprice * (1 - l_discount)) AS total_revenue
      from lineitem
      where l_shipdate >= date '1993-05-01'
        and l_shipdate < date '1993-05-01' + interval '3' month
      group by l_suppkey) s2
where s_suppkey = supplier_no and total_revenue = 1882892.8342
order by s_suppkey;

--Query 16
select
	p_brand,
	p_type,
	p_size,
	count(distinct ps_suppkey) as supplier_cnt
from
	partsupp join part on p_partkey = ps_partkey
    left join (
        select
            s_suppkey
        from
            supplier
        where s_comment like '%Customer%Complaints%'
    ) s on s.s_suppkey = ps_suppkey
where p_brand <> 'Brand#24'
	and p_type not like 'LARGE BRUSHED%'
	and p_size in (42, 19, 15, 18, 14, 27, 39, 30)
	and s.s_suppkey is NULL
group by
	p_brand,
	p_type,
	p_size
order by
	supplier_cnt desc,
	p_brand,
	p_type,
	p_size;

--Query 17
select
	sum(l_extendedprice) / 7.0 as avg_yearly
from
	lineitem,
	part,
    (
		select l_partkey, 0.2 * avg(l_quantity) as avg_fifth
		from
			lineitem
		group by l_partkey
	) s
where p_partkey = lineitem.l_partkey
	and p_brand = 'Brand#44'
	and p_container = 'WRAP BOX'
	and l_quantity < avg_fifth
	and p_partkey = s.l_partkey;

--Query 18
select
	c_name,
	c_custkey,
	o_orderkey,
	o_orderdate,
	o_totalprice,
	sum(l_quantity)
from
	customer,
	orders,
	lineitem
where o_orderkey in (
		select
			l_orderkey
		from
			lineitem
		group by
			l_orderkey having
				sum(l_quantity) > 314
	)
	and c_custkey = o_custkey
	and o_orderkey = l_orderkey
group by
	c_name,
	c_custkey,
	o_orderkey,
	o_orderdate,
	o_totalprice
order by
	o_totalprice desc,
	o_orderdate
limit 100;

--Query 19
select
	sum(l_extendedprice* (1 - l_discount)) as revenue
from
	 lineitem,
	 part
where p_partkey = l_partkey and (
	( p_brand = 'Brand#52'
		and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
		and l_quantity >= 8 and l_quantity <= 8 + 10
		and p_size between 1 and 5
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	( p_brand = 'Brand#22'
		and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
		and l_quantity >= 15 and l_quantity <= 15 + 10
		and p_size between 1 and 10
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	( p_brand = 'Brand#23'
		and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
		and l_quantity >= 28 and l_quantity <= 28 + 10
		and p_size between 1 and 15
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	));

--Query 20
select
	s_name,
	s_address
from
	supplier,
	nation
where s_suppkey in (
		select
			ps_suppkey
		from
			partsupp,
		    (
				select l_partkey, l_suppkey, 0.5 * sum(l_quantity) as avghalf
				from
					lineitem
				where l_shipdate >= date '1994-01-01'
					and l_shipdate < date '1994-01-01' + interval '1' year
		        group by l_partkey, l_suppkey
			) s
		where ps_partkey in (
				select
					p_partkey
				from
					part
				where p_name like 'goldenrod%'
			)
			and ps_availqty > s.avghalf
			and s.l_partkey = ps_partkey and s.l_suppkey = ps_suppkey
	)
	and s_nationkey = n_nationkey
	and n_name = 'GERMANY'
order by
	s_name;

--Query 21
select
	s_name,
	count(*) as numwait
from
	supplier,
	lineitem l1,
	orders,
	nation
where s_suppkey = l1.l_suppkey
	and o_orderkey = l1.l_orderkey
	and o_orderstatus = 'F'
	and l1.l_receiptdate > l1.l_commitdate
	and exists (
		select
			*
		from
			lineitem l2
		where l2.l_orderkey = l1.l_orderkey
			and l2.l_suppkey <> l1.l_suppkey
	)
	and not exists (
		select
			*
		from
			lineitem l3
		where l3.l_orderkey = l1.l_orderkey
			and l3.l_suppkey <> l1.l_suppkey
			and l3.l_receiptdate > l3.l_commitdate
	)
	and s_nationkey = n_nationkey
	and n_name = 'FRANCE'
group by
	s_name
order by
	numwait desc,
	s_name;

--Query 22
select
	cntrycode,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		select substring(c_phone, 1, 2) as cntrycode,
			c_acctbal
		from
			customer
		where substring(c_phone, 1, 2) in
				('27', '26', '32', '39', '35', '40', '29')
			and c_acctbal > 5008.25
			and not exists (
				select
					*
				from
					orders
				where o_custkey = c_custkey
			)
	) as custsale
group by
	cntrycode
order by
	cntrycode;
