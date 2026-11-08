# 데이터셋 정보 (자동 생성)

`python dataset_info.py`가 `cache/<ds>.npz`에서 만든다. 모든 실험이 실제로 쓰는 데이터와 같다.
컬럼 규칙: `configs/datasets.yaml`, 처리 방식: ARCHITECTURE.md §3.

| 데이터셋 | Tier | train | test | 숫자 피처 | 범주 피처 (원-핫 폭) | 클래스 | benign % | 불균형 (최대/최소) | 100행 미만 클래스 | 드롭 컬럼 | K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mirai | A | 40,893 | 9,335 | 26 | 1 (3) | 4 | 94.4 | 354 | 0 | 23 | 3 |
| nsl-kdd | A | 125,973 | 22,544 | 37 | 3 (84) | 23 | 53.5 | 33,671 | 12 | 4 | 4 |
| unsw-nb15 | A | 175,341 | 82,332 | 31 | 3 (155) | 10 | 31.9 | 430 | 0 | 2 | 6 |
| ton-iot | A | 810,485 | 347,509 | 6 | 2 (97) | 10 | 17.2 | 3,043 | 0 | 4 | 5 |
| cicids2017 | A | 1,979,286 | 848,590 | 69 | 0 (0) | 15 | 80.3 | 198,730 | 3 | 11 | 5 |
| xiiotid | A | 656,668 | 164,166 | 54 | 3 (23) | 10 | 51.3 | 906 | 0 | 6 | 7 |
| wustl-iiot-2021 | A | 955,571 | 238,893 | 40 | 1 (8) | 5 | 92.7 | 5,181 | 0 | 4 | 5 |
| edge-iiot | A | 1,775,360 | 443,841 | 36 | 7 (49) | 15 | 72.8 | 1,585 | 0 | 9 | 6 |
| cicids2018 | B | 11,294,033 | 4,843,150 | 68 | 1 (3) | 15 | 83.0 | 156,177 | 1 | 12 | 4 |
| lspr23 | B | 12,268,365 | 3,066,360 | 77 | 5 (92) | 14 | 95.3 | 2,339,346 | 4 | 12 | 5 |
| nf-ton-iot-v3 | B | 22,016,208 | 5,504,052 | 45 | 2 (138) | 10 | 61.0 | 4,203 | 0 | 4 | 6 |

- 클래스 수는 benign 포함. 불균형 = train에서 가장 큰 클래스 ÷ 가장 작은 클래스.
- 범주 피처의 원-핫 폭은 train에서 본 값의 수(+ 처음 보는 값 슬롯 1개는 제외).

## mirai

- 위치: `/home/work/yunuikang/multi-tester/data/mirai/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 3
- 행: train 40,893 / test 9,335
- 사용 피처: 숫자 26개 + 범주 1개 (`protocol` 3종)
- 드롭 — 누수(포트·날짜·난이도): `src_port`, `dst_port`
- 드롭 — 메타데이터(IP·시각·ID): `window_start`, `window_end`, `src_ip_addr`, `dst_ip_addr`
- 드롭 — 상수(train 고유값 1개): `bpkts_per_second`, `total_fhlen`, `backward_iat_std`, `backward_packet_length_mean`, `backward_iat_total`, `backward_packet_length_std`, `backward_packet_length_min`, `flow_cwr`, `flow_rst`, `backward_iat_mean`, `backward_packet_length_max`, `total_length_of_backward_packets`, `total_backward_packets`, `backward_iat_min`, `backward_iat_max`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `flow_iat_std`, `forward_packet_length_max`, `total_bhlen`, `flow_protocol`, `flow_iat_total`, `forward_iat_max`, `flow_iat_mean`, `forward_packet_length_mean`, `fpkts_per_second`, `flow_packets_per_second`, `forward_iat_total`, `forward_packet_length_std`, `forward_iat_min`, `total_forward_packets`, `flow_ece`, `forward_packet_length_min`, `flow_urg`, `flow_psh`, `forward_iat_std`, `total_length_of_forward_packets`, `forward_iat_mean`, `flow_fin`, `flow_ack`, `flow_syn`, `flow_iat_min`, `flow_iat_max`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 38,586 | 94.358 | 8,587 |
| 1 | port scanning (reconnaissance) | 1,617 | 3.954 | 324 |
| 2 | udp flooding (action) | 581 | 1.421 | 225 |
| 3 | bruteforce (infection) | 109 | 0.267 | 199 |

## nsl-kdd

- 위치: `/home/work/yunuikang/multi-tester/data/nsl-kdd/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 4
- 행: train 125,973 / test 22,544 (test 중 unknown 3,750행 = test에만 있는 라벨 17종: apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm)
- 사용 피처: 숫자 37개 + 범주 3개 (`protocol` 3종, `service` 70종, `flag` 11종)
- 드롭 — 누수(포트·날짜·난이도): `unknown`
- 드롭 — 상수(train 고유값 1개): `num_outbound_cmds`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `duration`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`, `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `is_host_login`, `is_guest_login`, `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate`, `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 67,343 | 53.458 | 9,711 |
| 1 | neptune | 41,214 | 32.717 | 4,657 |
| 2 | satan | 3,633 | 2.884 | 735 |
| 3 | ipsweep | 3,599 | 2.857 | 141 |
| 4 | portsweep | 2,931 | 2.327 | 157 |
| 5 | smurf | 2,646 | 2.100 | 665 |
| 6 | nmap | 1,493 | 1.185 | 73 |
| 7 | back | 956 | 0.759 | 359 |
| 8 | teardrop | 892 | 0.708 | 12 |
| 9 | warezclient | 890 | 0.707 | 0 |
| 10 | pod | 201 | 0.160 | 41 |
| 11 | guess_passwd | 53 | 0.042 | 1,231 |
| 12 | buffer_overflow | 30 | 0.024 | 20 |
| 13 | warezmaster | 20 | 0.016 | 944 |
| 14 | land | 18 | 0.014 | 7 |
| 15 | imap | 11 | 0.009 | 1 |
| 16 | rootkit | 10 | 0.008 | 13 |
| 17 | loadmodule | 9 | 0.007 | 2 |
| 18 | ftp_write | 8 | 0.006 | 3 |
| 19 | multihop | 7 | 0.006 | 18 |
| 20 | phf | 4 | 0.003 | 2 |
| 21 | perl | 3 | 0.002 | 2 |
| 22 | spy | 2 | 0.002 | 0 |
| −1 | unknown (test에만 있는 라벨) | 0 | 0 | 3,750 |

## unsw-nb15

- 위치: `/home/work/yunuikang/multi-tester/data/unsw-nb15/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 6
- 행: train 175,341 / test 82,332
- 사용 피처: 숫자 31개 + 범주 3개 (`protocol` 133종, `service` 13종, `state` 9종)
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `is_sm_ips_ports`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 56,000 | 31.938 | 37,000 |
| 1 | Generic | 40,000 | 22.813 | 18,871 |
| 2 | Exploits | 33,393 | 19.045 | 11,132 |
| 3 | Fuzzers | 18,184 | 10.371 | 6,062 |
| 4 | DoS | 12,264 | 6.994 | 4,089 |
| 5 | Reconnaissance | 10,491 | 5.983 | 3,496 |
| 6 | Analysis | 2,000 | 1.141 | 677 |
| 7 | Backdoor | 1,746 | 0.996 | 583 |
| 8 | Shellcode | 1,133 | 0.646 | 378 |
| 9 | Worms | 130 | 0.074 | 44 |

## ton-iot

- 위치: `/home/work/yunuikang/multi-tester/data/ton-iot/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 5
- 행: train 810,485 / test 347,509
- 사용 피처: 숫자 6개 + 범주 2개 (`protocol` 5종, `l7_proto` 92종)
- 드롭 — 누수(포트·날짜·난이도): `l4_src_port`, `l4_dst_port`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `in_bytes`, `out_bytes`, `in_pkts`, `out_pkts`, `tcp_flags`, `flow_duration_milliseconds`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 139,097 | 17.162 | 59,353 |
| 1 | injection | 322,596 | 39.803 | 138,216 |
| 2 | ddos | 138,197 | 17.051 | 59,483 |
| 3 | password | 101,203 | 12.487 | 43,589 |
| 4 | xss | 69,888 | 8.623 | 30,025 |
| 5 | scanning | 14,491 | 1.788 | 6,127 |
| 6 | backdoor | 12,063 | 1.488 | 5,180 |
| 7 | dos | 11,940 | 1.473 | 5,116 |
| 8 | mitm | 904 | 0.112 | 384 |
| 9 | ransomware | 106 | 0.013 | 36 |

## cicids2017

- 위치: `/home/work/yunuikang/multi-tester/data/cicids2017/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 5 · 라벨 통일: cicids2017
- 행: train 1,979,286 / test 848,590
- 사용 피처: 숫자 69개 + 범주 0개
- 드롭 — 누수(포트·날짜·난이도): `destination port`
- 드롭 — 상수(train 고유값 1개): `bwd psh flags`, `bwd urg flags`, `fwd avg bytes/bulk`, `fwd avg packets/bulk`, `fwd avg bulk rate`, `bwd avg bytes/bulk`, `bwd avg packets/bulk`, `bwd avg bulk rate`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `flow duration`, `total fwd packets`, `total backward packets`, `total length of fwd packets`, `total length of bwd packets`, `fwd packet length max`, `fwd packet length min`, `fwd packet length mean`, `fwd packet length std`, `bwd packet length max`, `bwd packet length min`, `bwd packet length mean`, `bwd packet length std`, `flow bytes/s`, `flow packets/s`, `flow iat mean`, `flow iat std`, `flow iat max`, `flow iat min`, `fwd iat total`, `fwd iat mean`, `fwd iat std`, `fwd iat max`, `fwd iat min`, `bwd iat total`, `bwd iat mean`, `bwd iat std`, `bwd iat max`, `bwd iat min`, `fwd psh flags`, `fwd urg flags`, `fwd header length`, `bwd header length`, `fwd packets/s`, `bwd packets/s`, `min packet length`, `max packet length`, `packet length mean`, `packet length std`, `packet length variance`, `fin flag count`, `syn flag count`, `rst flag count`, `psh flag count`, `ack flag count`, `urg flag count`, `cwe flag count`, `ece flag count`, `down/up ratio`, `average packet size`, `avg fwd segment size`, `avg bwd segment size`, `fwd header length__dup1`, `subflow fwd packets`, `subflow fwd bytes`, `subflow bwd packets`, `subflow bwd bytes`, `init_win_bytes_forward`, `init_win_bytes_backward`, `act_data_pkt_fwd`, `min_seg_size_forward`, `active mean`, `active std`, `active max`, `active min`, `idle mean`, `idle std`, `idle max`, `idle min`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 1,589,840 | 80.324 | 681,480 |
| 1 | dos hulk | 161,063 | 8.137 | 69,061 |
| 2 | portscan | 111,284 | 5.622 | 47,520 |
| 3 | ddos | 89,410 | 4.517 | 38,615 |
| 4 | dos goldeneye | 7,238 | 0.366 | 3,055 |
| 5 | ftp-patator | 5,481 | 0.277 | 2,454 |
| 6 | ssh-patator | 4,105 | 0.207 | 1,792 |
| 7 | dos slowloris | 4,038 | 0.204 | 1,758 |
| 8 | dos slowhttptest | 3,912 | 0.198 | 1,587 |
| 9 | bot | 1,364 | 0.069 | 592 |
| 10 | web attack brute force | 1,035 | 0.052 | 472 |
| 11 | web attack xss | 467 | 0.024 | 185 |
| 12 | infiltration | 29 | 0.001 | 7 |
| 13 | web attack sql injection | 12 | 0.001 | 9 |
| 14 | heartbleed | 8 | 0.000 | 3 |

## xiiotid

- 위치: `/home/work/yunuikang/multi-tester/data/xiiotid/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 7
- 행: train 656,668 / test 164,166
- 사용 피처: 숫자 54개 + 범주 3개 (`protocol` 4종, `service` 17종, `flag` 2종)
- 드롭 — 누수(포트·날짜·난이도): `Scr_port`, `Des_port`
- 드롭 — 상수(train 고유값 1개): `Bad_checksum`, `is_SYN_with_RST`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `Duration`, `Scr_bytes`, `Des_bytes`, `missed_bytes`, `is_syn_only`, `Is_SYN_ACK`, `is_pure_ack`, `is_with_payload`, `FIN or RST`, `Scr_pkts`, `Scr_ip_bytes`, `Des_pkts`, `Des_ip_bytes`, `anomaly_alert`, `total_bytes`, `total_packet`, `paket_rate`, `byte_rate`, `Scr_packts_ratio`, `Des_pkts_ratio`, `Scr_bytes_ratio`, `Des_bytes_ratio`, `Avg_user_time`, `Std_user_time`, `Avg_nice_time`, `Std_nice_time`, `Avg_system_time`, `Std_system_time`, `Avg_iowait_time`, `Std_iowait_time`, `Avg_ideal_time`, `Std_ideal_time`, `Avg_tps`, `Std_tps`, `Avg_rtps`, `Std_rtps`, `Avg_wtps`, `Std_wtps`, `Avg_ldavg_1`, `Std_ldavg_1`, `Avg_kbmemused`, `Std_kbmemused`, `Avg_num_Proc/s`, `Std_num_proc/s`, `Avg_num_cswch/s`, `std_num_cswch/s`, `OSSEC_alert`, `OSSEC_alert_level`, `Login_attempt`, `Succesful_login`, `File_activity`, `Process_activity`, `read_write_physical.process`, `is_privileged`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 337,134 | 51.340 | 84,283 |
| 1 | RDOS | 113,233 | 17.244 | 28,028 |
| 2 | Reconnaissance | 101,763 | 15.497 | 25,827 |
| 3 | Weaponization | 53,886 | 8.206 | 13,374 |
| 4 | Lateral _movement | 25,245 | 3.844 | 6,351 |
| 5 | Exfiltration | 17,706 | 2.696 | 4,428 |
| 6 | Tampering | 4,108 | 0.626 | 1,014 |
| 7 | C&C | 2,313 | 0.352 | 550 |
| 8 | Exploitation | 908 | 0.138 | 225 |
| 9 | crypto-ransomware | 372 | 0.057 | 86 |

## wustl-iiot-2021

- 위치: `/home/work/yunuikang/multi-tester/data/wustl-iiot-2021/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 5
- 행: train 955,571 / test 238,893
- 사용 피처: 숫자 40개 + 범주 1개 (`protocol` 8종)
- 드롭 — 누수(포트·날짜·난이도): `Sport`, `Dport`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `Mean`, `SrcPkts`, `DstPkts`, `TotPkts`, `DstBytes`, `SrcBytes`, `TotBytes`, `SrcLoad`, `DstLoad`, `Load`, `SrcRate`, `DstRate`, `Rate`, `SrcLoss`, `DstLoss`, `Loss`, `pLoss`, `SrcJitter`, `DstJitter`, `SIntPkt`, `DIntPkt`, `Dur`, `TcpRtt`, `IdleTime`, `Sum`, `Min`, `Max`, `sDSb`, `sTtl`, `dTtl`, `sIpId`, `dIpId`, `SAppBytes`, `DAppBytes`, `TotAppByte`, `SynAck`, `RunTime`, `sTos`, `SrcJitAct`, `DstJitAct`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 885,958 | 92.715 | 221,490 |
| 1 | DoS | 62,632 | 6.554 | 15,673 |
| 2 | Reconn | 6,601 | 0.691 | 1,639 |
| 3 | CommInj | 209 | 0.022 | 50 |
| 4 | Backdoor | 171 | 0.018 | 41 |

## edge-iiot

- 위치: `/home/work/yunuikang/multi-tester/data/edge-iiot/{training,test}-flow.csv` · Tier A · Step1 K(elbow) = 6
- 행: train 1,775,360 / test 443,841
- 사용 피처: 숫자 36개 + 범주 7개 (`http.request.method` 8종, `http.referer` 4종, `http.request.version` 12종, `dns.qry.name.len` 9종, `mqtt.conack.flags` 12종, `mqtt.protoname` 2종, `mqtt.topic` 2종)
- 드롭 — 누수(포트·날짜·난이도): `http.tls_port`, `tcp.dstport`, `tcp.srcport`, `udp.port`
- 드롭 — 상수(train 고유값 1개): `icmp.unused`, `dns.qry.type`, `mqtt.msg_decoded_as`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `arp.opcode`, `arp.hw.size`, `icmp.checksum`, `icmp.seq_le`, `icmp.transmit_timestamp`, `http.content_length`, `http.response`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `tcp.connection.fin`, `tcp.connection.rst`, `tcp.connection.syn`, `tcp.connection.synack`, `tcp.flags`, `tcp.flags.ack`, `tcp.len`, `tcp.seq`, `udp.stream`, `udp.time_delta`, `dns.qry.name`, `dns.qry.qu`, `dns.retransmission`, `dns.retransmit_request`, `dns.retransmit_request_in`, `mqtt.conflag.cleansess`, `mqtt.conflags`, `mqtt.hdrflags`, `mqtt.len`, `mqtt.msgtype`, `mqtt.proto_len`, `mqtt.topic_len`, `mqtt.ver`, `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 1,292,514 | 72.803 | 323,129 |
| 1 | DDoS_UDP | 96,982 | 5.463 | 24,586 |
| 2 | DDoS_ICMP | 93,380 | 5.260 | 23,056 |
| 3 | SQL_injection | 41,026 | 2.311 | 10,177 |
| 4 | DDoS_TCP | 40,217 | 2.265 | 9,845 |
| 5 | Password | 40,152 | 2.262 | 10,001 |
| 6 | Vulnerability_scanner | 40,108 | 2.259 | 10,002 |
| 7 | DDoS_HTTP | 39,826 | 2.243 | 10,085 |
| 8 | Uploading | 30,066 | 1.694 | 7,568 |
| 9 | Backdoor | 19,876 | 1.120 | 4,986 |
| 10 | Port_Scanning | 17,974 | 1.012 | 4,590 |
| 11 | XSS | 12,655 | 0.713 | 3,260 |
| 12 | Ransomware | 8,823 | 0.497 | 2,102 |
| 13 | MITM | 946 | 0.053 | 268 |
| 14 | Fingerprinting | 815 | 0.046 | 186 |

## cicids2018

- 위치: `/home/work/yunuikang/multi-tester/data/cicids2018/{training,test}-flow.csv` · Tier B · Step1 K(elbow) = 4
- 행: train 11,294,033 / test 4,843,150
- 사용 피처: 숫자 68개 + 범주 1개 (`protocol` 3종)
- 드롭 — 누수(포트·날짜·난이도): `dst port`
- 드롭 — 메타데이터(IP·시각·ID): `timestamp`
- 드롭 — 상수(train 고유값 1개): `bwd psh flags`, `bwd urg flags`, `fwd byts/b avg`, `fwd pkts/b avg`, `fwd blk rate avg`, `bwd byts/b avg`, `bwd pkts/b avg`, `bwd blk rate avg`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `flow duration`, `tot fwd pkts`, `tot bwd pkts`, `totlen fwd pkts`, `totlen bwd pkts`, `fwd pkt len max`, `fwd pkt len min`, `fwd pkt len mean`, `fwd pkt len std`, `bwd pkt len max`, `bwd pkt len min`, `bwd pkt len mean`, `bwd pkt len std`, `flow byts/s`, `flow pkts/s`, `flow iat mean`, `flow iat std`, `flow iat max`, `flow iat min`, `fwd iat tot`, `fwd iat mean`, `fwd iat std`, `fwd iat max`, `fwd iat min`, `bwd iat tot`, `bwd iat mean`, `bwd iat std`, `bwd iat max`, `bwd iat min`, `fwd psh flags`, `fwd urg flags`, `fwd header len`, `bwd header len`, `fwd pkts/s`, `bwd pkts/s`, `pkt len min`, `pkt len max`, `pkt len mean`, `pkt len std`, `pkt len var`, `fin flag cnt`, `syn flag cnt`, `rst flag cnt`, `psh flag cnt`, `ack flag cnt`, `urg flag cnt`, `cwe flag count`, `ece flag cnt`, `down/up ratio`, `pkt size avg`, `fwd seg size avg`, `bwd seg size avg`, `subflow fwd pkts`, `subflow fwd byts`, `subflow bwd pkts`, `subflow bwd byts`, `init fwd win byts`, `init bwd win byts`, `fwd act data pkts`, `fwd seg size min`, `active mean`, `active std`, `active max`, `active min`, `idle mean`, `idle std`, `idle max`, `idle min`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 9,370,620 | 82.970 | 4,019,629 |
| 1 | ddos hoic | 480,527 | 4.255 | 205,485 |
| 2 | ddos loic-http | 403,762 | 3.575 | 172,429 |
| 3 | dos hulk | 322,987 | 2.860 | 138,925 |
| 4 | bot | 200,372 | 1.774 | 85,819 |
| 5 | ftp-patator | 135,785 | 1.202 | 57,569 |
| 6 | ssh-patator | 130,806 | 1.158 | 56,783 |
| 7 | infilteration | 112,518 | 0.996 | 48,121 |
| 8 | dos slowhttptest | 97,920 | 0.867 | 41,970 |
| 9 | dos goldeneye | 29,125 | 0.258 | 12,383 |
| 10 | dos slowloris | 7,781 | 0.069 | 3,209 |
| 11 | ddos loic-udp | 1,178 | 0.010 | 552 |
| 12 | brute force | 430 | 0.004 | 181 |
| 13 | xss | 162 | 0.001 | 68 |
| 14 | sql injection | 60 | 0.001 | 27 |

## lspr23

- 위치: `/home/work/yunuikang/multi-tester/data/lspr23/{training,test}-flow.csv` · Tier B · Step1 K(elbow) = 5
- 행: train 12,268,365 / test 3,066,360
- 사용 피처: 숫자 77개 + 범주 5개 (`protocol` 8종, `flag` 12종, `service` 29종, `Segment_src` 22종, `Segment_dst` 21종)
- 드롭 — 누수(포트·날짜·난이도): `SrcPort`, `DstPort`
- 드롭 — 메타데이터(IP·시각·ID): `Int/Ext Dst IP`
- 드롭 — 상수(train 고유값 1개): `Bwd PSH flags`, `Fwd URG flags`, `Bwd URG flags`, `URG Flag Cnt`, `Fwd Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd Bulk Rate Avg`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `Flow Duration`, `Flow Bytes/s`, `Flow Packets/s`, `Tot Fwd Pkts`, `Tot Bwd Pkts`, `Total Length of Fwd Packet`, `Total Length of Bwd Packet`, `Fwd Packet Length Min`, `Fwd Packet Length Max`, `Fwd Packet Length Mean`, `Fwd Packet Length Std`, `Bwd Packet Length Min`, `Bwd Packet Length Max`, `Bwd Packet Length Mean`, `Bwd Packet Length Std`, `Flow IAT Mean`, `Flow IAT Min`, `Flow IAT Max`, `Flow IAT Stddev`, `Fwd IAT Min`, `Fwd IAT Max`, `Fwd IAT Mean`, `Fwd IAT Std`, `Fwd IAT Tot`, `Bwd IAT Min`, `Bwd IAT Max`, `Bwd IAT Mean`, `Bwd IAT Std`, `Bwd IAT Tot`, `Fwd PSH flags`, `Fwd Header Length`, `Bwd Header Length`, `Fwd Packets/s`, `Bwd Packets/s`, `Packet Length Min`, `Packet Length Max`, `Packet Length Mean`, `Packet Length Std`, `Packet Length Variance`, `FIN Flag Cnt`, `SYN Flag Cnt`, `RST Flag Cnt`, `PSH Flag Cnt`, `ACK Flag Cnt`, `CWR Flag Cnt`, `ECE Flag Cnt`, `Down/Up Ratio`, `Average Packet Size`, `Fwd Segment Size Avg`, `Bwd Segment Size Avg`, `Bwd Bytes/Bulk Avg`, `Bwd Packet/Bulk Avg`, `Bwd Bulk Rate Avg`, `Subflow Fwd Packets`, `Subflow Fwd Bytes`, `Subflow Bwd Packets`, `Subflow Bwd Bytes`, `FWD Init Win Bytes`, `Bwd Init Win Bytes`, `Fwd Act Data Pkts`, `Fwd Seg Size Min`, `Active Min`, `Active Mean`, `Active Max`, `Active Std`, `Idle Min`, `Idle Mean`, `Idle Max`, `Idle Std`, `SigID revision`, `Severity`, `Anomaly_event`, `L3/L4 Protocol`, `Label_src`, `Label_dst`, `External_src`, `External_dst`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 11,696,731 | 95.341 | 2,923,984 |
| 1 | generic protocol command decode | 526,228 | 4.289 | 131,016 |
| 2 | misc activity | 37,025 | 0.302 | 9,292 |
| 3 | attempted information leak | 4,394 | 0.036 | 1,119 |
| 4 | potentially bad traffic | 1,856 | 0.015 | 421 |
| 5 | potential corporate privacy violation | 885 | 0.007 | 221 |
| 6 | network trojan detected | 426 | 0.003 | 124 |
| 7 | network scan | 220 | 0.002 | 49 |
| 8 | unwanted program detected | 215 | 0.002 | 47 |
| 9 | attempted denial of service | 187 | 0.002 | 40 |
| 10 | device retrieving external ip | 91 | 0.001 | 21 |
| 11 | not suspicious traffic | 82 | 0.001 | 19 |
| 12 | web application attack | 20 | 0.000 | 4 |
| 13 | attempted user privilege gain | 5 | 0.000 | 3 |

## nf-ton-iot-v3

- 위치: `/home/work/yunuikang/multi-tester/data/nf-ton-iot-v3/{training,test}-flow.csv` · Tier B · Step1 K(elbow) = 6
- 행: train 22,016,208 / test 5,504,052
- 사용 피처: 숫자 45개 + 범주 2개 (`protocol` 5종, `L7_PROTO` 133종)
- 드롭 — 누수(포트·날짜·난이도): `L4_SRC_PORT`, `L4_DST_PORT`
- 드롭 — 다른 라벨 컬럼: `attack_flag`, `attack_step`
- 숫자 피처 목록: `IN_BYTES`, `IN_PKTS`, `OUT_BYTES`, `OUT_PKTS`, `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT`, `MIN_TTL`, `MAX_TTL`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`, `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`, `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `FTP_COMMAND_RET_CODE`, `SRC_TO_DST_IAT_MIN`, `SRC_TO_DST_IAT_MAX`, `SRC_TO_DST_IAT_AVG`, `SRC_TO_DST_IAT_STDDEV`, `DST_TO_SRC_IAT_MIN`, `DST_TO_SRC_IAT_MAX`, `DST_TO_SRC_IAT_AVG`, `DST_TO_SRC_IAT_STDDEV`

| # | 클래스 | train | train % | test |
|---:|---|---:|---:|---:|
| 0 | benign | 13,433,771 | 61.018 | 3,358,443 |
| 1 | ddos | 3,312,440 | 15.045 | 828,816 |
| 2 | xss | 2,267,668 | 10.300 | 566,767 |
| 3 | password | 1,276,710 | 5.799 | 318,067 |
| 4 | scanning | 1,086,952 | 4.937 | 272,025 |
| 5 | injection | 305,575 | 1.388 | 76,202 |
| 6 | Backdoor | 162,629 | 0.739 | 40,755 |
| 7 | dos | 162,449 | 0.738 | 41,007 |
| 8 | mitm | 4,818 | 0.022 | 1,195 |
| 9 | ransomware | 3,196 | 0.015 | 775 |
